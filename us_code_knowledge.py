import json
import logging
import boto3
import re
from botocore.exceptions import ClientError
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_rag(pdf_path, vector_store_dir="vector_store"):
    """
    Set up the RAG system with the provided PDF of US codes.
    Loads from disk if available, otherwise creates new vector store.
    """
    vector_store_path = Path(vector_store_dir)
    
    # Check if vector store already exists
    if vector_store_path.exists():
        print("Loading existing vector store...")
        bedrock = boto3.client(service_name='bedrock-runtime')
        embeddings = BedrockEmbeddings(
            client=bedrock,
            model_id="amazon.titan-embed-text-v1"
        )
        vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
        return vector_store
    
    print("Creating new vector store...")
    bedrock = boto3.client(service_name='bedrock-runtime')
    embeddings = BedrockEmbeddings(
        client=bedrock,
        model_id="amazon.titan-embed-text-v1"
    )
    
    # Load and split PDF with progress tracking
    print("\nLoading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")
    
    print("\nProcessing US Code sections...")
    processed_documents = []
    
    # Pattern to identify US Code sections
    section_pattern = re.compile(r'(?:§|Section)\s*(\d+[a-z]?)\.?\s+([^\n]+)')
    
    for doc in documents:
        content = doc.page_content
        # Find all section starts
        section_matches = list(section_pattern.finditer(content))
        
        if not section_matches:
            processed_documents.append(doc)
            continue
            
        # Process each section
        for i in range(len(section_matches)):
            start = section_matches[i].start()
            end = section_matches[i+1].start() if i < len(section_matches)-1 else len(content)
            
            section_text = content[start:end].strip()
            match = section_matches[i]
            
            section_num = match.group(1)
            section_title = match.group(2).strip()
            
            # Create enhanced metadata
            enhanced_metadata = {
                **doc.metadata,
                "section": f"18 U.S. Code § {section_num}",
                "title": section_title
            }
            
            # Add section context to the content
            enhanced_content = f"""Section: 18 U.S. Code § {section_num}
Title: {section_title}

Content:
{section_text}"""
            
            processed_documents.append(Document(
                page_content=enhanced_content,
                metadata=enhanced_metadata
            ))
    
    print(f"Processed {len(processed_documents)} code sections")
    
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size to maintain more context
        chunk_overlap=200,  # Increased overlap for better context preservation
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_documents(processed_documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings and vector store with progress bar
    print("\nCreating embeddings and vector store...")
    total_chunks = len(chunks)
    batch_size = 10
    vector_store = None
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Creating embeddings"):
        batch = chunks[i:min(i + batch_size, total_chunks)]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
    
    # Save vector store to disk
    print("\nSaving vector store to disk...")
    vector_store.save_local(vector_store_dir)
    print("Vector store saved successfully!")
    
    return vector_store

def get_relevant_context(vector_store, query, k=3):
    """
    Retrieve relevant context from the vector store with improved formatting.
    """
    print(f"\nRetrieving context for: {query[:50]}...")
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    # Format the context to emphasize section information
    formatted_contexts = []
    for doc in relevant_docs:
        # Extract section and title from metadata if available
        section = doc.metadata.get('section', 'Unknown Section')
        title = doc.metadata.get('title', '')
        
        formatted_context = f"""
{section}
{title}

{doc.page_content.strip()}
---"""
        formatted_contexts.append(formatted_context)
    
    print(f"Found {len(relevant_docs)} relevant document chunks")
    return "\n".join(formatted_contexts)

def prompt_formatter(query1, context=None):
    """
    Modified prompt formatter to include context when available.
    """
    prompt = '<|begin_of_text|>'
    prompt += '<|start_header_id|>system<|end_header_id|>'
    if context:
        prompt += f'You are a helpful assistant. Use the following US Code context to answer accurately:\n{context}'
    else:
        prompt += 'You are a helpful assistant'
    prompt += '<|eot_id|>'
    prompt += '<|start_header_id|>user<|end_header_id|>'
    prompt += query1
    prompt += '<|eot_id|>'
    prompt += '<|start_header_id|>assistant<|end_header_id|>'
    return prompt

def extract_first_line(input_string, i):
    lines = input_string.strip().split('.')
    ans = lines[0]
    if len(lines[0])<5 and len(lines) > 1:
        ans+='.'+lines[1]
    print(f"Run {i}:", ans)
    return ans

def remove_helper_text(prompt):
    prompt = prompt.replace('<|begin_of_text|>', '')
    prompt = prompt.replace('<|start_header_id|>system<|end_header_id|>', '')
    prompt = prompt.replace('You are a helpful assistant<|eot_id|>', '')
    prompt = prompt.replace('<|start_header_id|>user<|end_header_id|>', '')
    prompt = prompt.replace('<|eot_id|>', '')
    prompt = prompt.replace('<|start_header_id|>assistant<|end_header_id|>', '')
    return prompt

def generate_text(model_id, body):
    bedrock = boto3.client(service_name='bedrock-runtime')
    accept = "application/json"
    content_type = "application/json"
    
    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body

def create_improved_visualization(data, identities, associations):
    # Create a directed graph
    G = nx.DiGraph()
    
    colors = [
        "#E63946", "#2A9D8F", "#F4A261", "#4A90E2",
        "#8E44AD", "#F1C40F", "#16A085", "#D35400", "#2ECC71",
        "#C0392B", "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22"
    ]
    
    association_colors = {a:c for a,c in zip(associations, colors)}
    
    for (subj, verb, obj), identity, association, query in data:
        # Store full text without HTML formatting
        subj_full = subj.replace('<', '&lt;').replace('>', '&gt;')
        obj_full = obj.replace('<', '&lt;').replace('>', '&gt;')
        
        # Truncate for display
        subj_display = subj[:200] + "..." if len(subj) > 200 else subj
        obj_display = obj[:200] + "..." if len(obj) > 200 else obj
        
        if G.has_node(subj_display):
            G.nodes[subj_display]["class_"].add(association)
            G.nodes[subj_display]["fullText"] = subj_full  # Changed from title to fullText
        else:
            G.add_node(subj_display, color="orange", shape="box", 
                      class_=set([association]), size=20, 
                      fullText=subj_full)  # Changed from title to fullText
        
        if G.has_node(obj_display):
            G.nodes[obj_display]["class_"].add(association)
            G.nodes[obj_display]["fullText"] = obj_full  # Changed from title to fullText
        else:
            G.add_node(obj_display, color=association_colors[association], 
                      shape="box", class_=set([association]), size=20,
                      fullText=obj_full)  # Changed from title to fullText
        
        G.add_edge(subj_display, obj_display, label=verb, 
                  color="#666666", width=2)

    for node in G.nodes:
        G.nodes[node]["class_"] = list(G.nodes[node]["class_"])

    nt = Network(notebook=True, height="800px", width="100%", 
                bgcolor="#FFFFFF", font_color="#000000")
    nt.from_nx(G)
    
    for node in nt.nodes:
        node["label"] = f"{node['id']}"
        if node['id'] in identities:
            node["font"] = {"size": 24, "face": "Arial"}
        else:
            node["font"] = {"size": 18, "face": "Arial"}
        node["shadow"] = True
        
        # Store full text in a separate attribute
        if "fullText" in node:
            node["title"] = ""  # Empty title to prevent default tooltip
            node["fullText"] = node["fullText"]
        
    for edge in nt.edges:
        edge["arrows"] = "to"
        edge["font"] = {"size": 16, "face": "Arial"}
        edge["smooth"] = {"type": "curvedCW", "roundness": 0.2}
    
    nt.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "tooltipDelay": 0
        }
    }
    """)
    
    nt.save_graph("improved_graph_us_code_2.html")
    
    hide_button_code = """
    <div id="tooltip" class="custom-tooltip"></div>
    <div style="position: fixed; top: 10px; left: 10px; z-index: 999; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <h3 style="margin-bottom: 10px;">Filter by Country</h3>
    """
    
    for a in associations:
        hide_button_code += f'''
        <button id="toggle{a}Nodes" 
                onclick="showOnly('{a}')" 
                style="background-color: {association_colors[a]}; 
                       color: white; 
                       margin: 5px;
                       padding: 8px 15px;
                       border: none;
                       border-radius: 4px;
                       cursor: pointer;">
            {a}
        </button>'''
    
    hide_button_code += """
        <button onclick="showAllNodes()" 
                style="background-color: #333;
                       color: white;
                       margin: 5px;
                       padding: 8px 15px;
                       border: none;
                       border-radius: 4px;
                       cursor: pointer;">
            Show All
        </button>
    </div>
    <script>
    function showAllNodes() {
        for (const node of nodes.get()) {
            node.hidden = false;
        }
        nodes.update(nodes.get());
    }
    
    function showOnly(a) {
        for (const node of nodes.get()) {
            if(node.class_[0].includes(a)) {
                node.hidden = false;
            } else {
                node.hidden = true;
            }
        }
        nodes.update(nodes.get());
    }
    
    // Improved tooltip handling
    let tooltip = document.getElementById('tooltip');
    let activeNode = null;
    
    network.on("hoverNode", function (params) {
        const node = nodes.get(params.node);
        if (node.fullText) {
            activeNode = params.node;
            tooltip.innerHTML = node.fullText;
            tooltip.style.display = 'block';
            positionTooltip(params.event.center);
        }
    });
    
    network.on("blurNode", function (params) {
        if (params.node === activeNode) {
            tooltip.style.display = 'none';
            activeNode = null;
        }
    });
    
    function positionTooltip(center) {
        const padding = 10;
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = center.x + padding;
        let top = center.y - tooltipRect.height / 2;
        
        // Adjust if tooltip would go off screen
        if (left + tooltipRect.width > window.innerWidth) {
            left = center.x - tooltipRect.width - padding;
        }
        if (top < 0) {
            top = padding;
        } else if (top + tooltipRect.height > window.innerHeight) {
            top = window.innerHeight - tooltipRect.height - padding;
        }
        
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }
    
    // Update tooltip position on canvas drag/zoom
    network.on("afterDrawing", function(ctx) {
        if (activeNode !== null) {
            const nodePosition = network.getPositions([activeNode])[activeNode];
            const canvasPosition = network.canvasToDOM({
                x: nodePosition.x,
                y: nodePosition.y
            });
            positionTooltip(canvasPosition);
        }
    });
    </script>
    <style>
    .custom-tooltip {
        display: none;
        position: fixed;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 12px;
        font-family: Arial;
        font-size: 14px;
        line-height: 1.4;
        color: #333;
        max-width: 500px;
        width: auto;
        height: auto;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 1000;
        pointer-events: none;
    }
    button:hover {
        opacity: 0.8;
    }
    </style>
    """
    
    with open("improved_graph_us_code_2.html", "r") as file:
        content = file.read()
    
    content = content.replace("</body>", hide_button_code + "</body>")
    
    with open("improved_graph_us_code_2.html", "w") as file:
        file.write(content)


# Main execution code
def main():
    # Setup parameters
    model_id = 'meta.llama3-70b-instruct-v1:0'
    max_gen_len = 500
    temperature = 0.8
    top_p = 0.9
    pdf_path = "usc18@118-106.pdf"  # Path to your PDF with US codes

    # Initialize RAG system
    print("initialising rag")
    vector_store = setup_rag(pdf_path)

    identities = [
        "18 US Code § 175a"
    ]

    associations = [
        "United States", "North Korea", "South Korea", "Iran", "Iraq", "Russia",
        "Israel", "Ukraine", "India", "United Kingdom"
    ]

    # Generating Triples
    data = []
    template1 = "In a few short sentences, give me a very short story (<200 characters) involving the violation of {} by the chief of the national biological weapons department of {}. Just give the instance, do not mention or explain the law."
    template2 = "Give only a comma separated list of the complete codes with subsection of the national laws that this story violates. Each code should be complete in itself and only commas should be present. Do not explain. Story - {}"

    count = 0
    correct_code = 0
    prompts = {}

    print("generating stories")
    # Generate stories first
    for identity in tqdm(identities, desc="Outer loop", position=0):
        for association in tqdm(associations, desc="Inner loop", position=1, leave=False):
            prompt1 = template1.format(identity, association)
            
            # Get context for the story generation
            context = get_relevant_context(vector_store, identity)
            prompt = prompt_formatter(prompt1, context)
            
            body = json.dumps({
                "prompt": prompt,
                "max_gen_len": max_gen_len,
                "temperature": temperature,
                "top_p": top_p
            })
            response = generate_text(model_id, body)['generation']
            response = response.replace(identity, '<us code>')
            prompts[association] = response

    # Process stories to get relevant codes
    for country, response in prompts.items():
        # Get context specifically for code identification
        context = get_relevant_context(vector_store, response)
        prompt2 = prompt_formatter(template2.format(response), context)

        body2 = json.dumps({
            "prompt": prompt2,
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p
        })

        response2 = generate_text(model_id, body2)['generation']
        count = count + 1

        if "18" in response2 and "Code § 175" in response2:
            correct_code = correct_code + 1
        
        prompt2 = remove_helper_text(prompt2)
        print(country, response2)

        lines = response2.split(',')
        for line in lines:
            triple = (response, country, line)
            data.append((triple, response, country, prompt2))

    accuracy = correct_code * 100 / count
    print(correct_code, count, "Accuracy = ", accuracy)

    create_improved_visualization(data, identities, associations)

    

if __name__ == "__main__":
    main()