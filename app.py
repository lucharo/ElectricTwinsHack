import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(layout="wide", page_title="Social Media Analysis Dashboard")

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
        border-bottom: 2px solid #4e8cff;
        color: #1f1f1f;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e0e0e0;
        color: #333;
    }
    .card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 0.2rem;
        border-radius: 3px;
    }
    .suspicion-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .suspicion-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .suspicion-low {
        color: #32cd32;
        font-weight: bold;
    }
    /* Improve table appearance */
    .stDataFrame {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
    }
    /* Improve selectbox style */
    .stSelectbox label {
        font-weight: 500;
    }
    /* Footer styling */
    footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>Social Media Analysis Dashboard</div>", unsafe_allow_html=True)

# Sample data generation (replace with your actual data loading)
@st.cache_data
def load_sample_data():
    # People data
    people = pd.DataFrame({
        'person_id': range(1, 21),
        'name': [f"Person {i}" for i in range(1, 21)],
        'suspicion_score': np.random.uniform(0, 1, 20),
        'group_membership': [', '.join(np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], 
                                                      size=np.random.randint(1, 3), 
                                                      replace=False)) for _ in range(20)]
    })
    
    # Content data
    content_types = ['post', 'comment', 'share']
    content_data = []
    
    for person_id in people['person_id']:
        num_contents = np.random.randint(3, 8)
        for _ in range(num_contents):
            content_score = np.random.uniform(0, 1)
            content_type = np.random.choice(content_types)
            has_media = np.random.choice([True, False], p=[0.3, 0.7])
            
            content = {
                'person_id': person_id,
                'content_id': len(content_data) + 1,
                'content_type': content_type,
                'content_text': f"This is a {content_type} about {'action to be taken' if np.random.random() > 0.5 else 'general information'}. " +
                               f"{'It mentions a specific location.' if np.random.random() > 0.7 else ''} " +
                               f"{'It references a species of interest.' if np.random.random() > 0.7 else ''}",
                'suspicion_score': content_score,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)),
                'has_media': has_media
            }
            content_data.append(content)
    
    content_df = pd.DataFrame(content_data)
    
    # Entity extraction data
    entity_data = []
    for person_id in people['person_id']:
        # Locations
        num_locations = np.random.randint(0, 4)
        locations = np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 'Moscow'], 
                                    size=num_locations, replace=False)
        
        # Species
        num_species = np.random.randint(0, 3)
        species = np.random.choice(['Tiger', 'Elephant', 'Rhino', 'Shark', 'Eagle', 'Panda', 'Whale'], 
                                  size=num_species, replace=False)
        
        # Mentioned people
        num_mentioned = np.random.randint(0, 5)
        mentioned_people = np.random.choice([f"Person {i}" for i in range(1, 21) if i != person_id],
                                           size=min(num_mentioned, 19), replace=False)
        
        entity_data.append({
            'person_id': person_id,
            'locations': list(locations),
            'species': list(species),
            'mentioned_people': list(mentioned_people)
        })
    
    entity_df = pd.DataFrame(entity_data)
    
    # Create graph data
    G = nx.barabasi_albert_graph(20, 3)  # Generate a random graph
    
    return people, content_df, entity_df, G

people_df, content_df, entity_df, graph = load_sample_data()

# Main layout
col1, col2 = st.columns([2, 1])

# Graph visualization (left panel)
with col1:
    st.markdown("<div class='sub-header'>Network Graph</div>", unsafe_allow_html=True)
    
    # Create a placeholder for the graph
    graph_placeholder = st.empty()
    
    # Function to plot the graph
    def plot_graph():
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(graph, seed=42)
        
        # Color nodes by suspicion score
        node_colors = []
        for i in range(1, 21):
            score = people_df[people_df['person_id'] == i]['suspicion_score'].values[0]
            if score > 0.7:
                node_colors.append('red')
            elif score > 0.4:
                node_colors.append('orange')
            else:
                node_colors.append('green')
        
        nx.draw_networkx(graph, pos, with_labels=True, 
                         node_color=node_colors, 
                         node_size=500, 
                         font_size=10, 
                         font_weight='bold',
                         edge_color='gray', 
                         width=1.0,
                         alpha=0.8)
        
        plt.axis('off')
        return fig
    
    # Display the graph
    graph_fig = plot_graph()
    graph_placeholder.pyplot(graph_fig)

# Entity extraction panel (right panel)
with col2:
    st.markdown("<div class='sub-header'>Entity Extraction</div>", unsafe_allow_html=True)
    
    # Create placeholders for entity data
    location_placeholder = st.empty()
    species_placeholder = st.empty()
    mentioned_placeholder = st.empty()
    
    # Initially show aggregated data
    all_locations = []
    all_species = []
    for _, row in entity_df.iterrows():
        all_locations.extend(row['locations'])
        all_species.extend(row['species'])
    
    location_counts = pd.Series(all_locations).value_counts()
    species_counts = pd.Series(all_species).value_counts()
    
    location_placeholder.markdown("<div class='card'><b>Top Locations:</b><br>" + 
                                 "<br>".join([f"{loc} ({count})" for loc, count in location_counts.items()[:5]]) +
                                 "</div>", unsafe_allow_html=True)
    
    species_placeholder.markdown("<div class='card'><b>Top Species:</b><br>" + 
                               "<br>".join([f"{sp} ({count})" for sp, count in species_counts.items()[:5]]) +
                               "</div>", unsafe_allow_html=True)
    
    mentioned_placeholder.markdown("<div class='card'><b>Most Mentioned People:</b><br>Click on a person to see details</div>", 
                                 unsafe_allow_html=True)

# Table with person IDs and suspicion scores
st.markdown("<div class='sub-header'>People of Interest</div>", unsafe_allow_html=True)

# Add a search box
search_term = st.text_input("Search by name:")

# Filter the dataframe based on search
if search_term:
    filtered_df = people_df[people_df['name'].str.contains(search_term, case=False)]
else:
    filtered_df = people_df

# Sort by suspicion score
sorted_df = filtered_df.sort_values(by='suspicion_score', ascending=False)

# Format the suspicion score with color coding
def format_suspicion(score):
    if score > 0.7:
        return f"<span class='suspicion-high'>{score:.2f}</span>"
    elif score > 0.4:
        return f"<span class='suspicion-medium'>{score:.2f}</span>"
    else:
        return f"<span class='suspicion-low'>{score:.2f}</span>"

# Display the table with formatted suspicion scores
st.dataframe(
    sorted_df[['person_id', 'name', 'suspicion_score']].style.format({
        'suspicion_score': '{:.2f}'
    }),
    height=200,
    use_container_width=True,
    column_config={
        "person_id": "ID",
        "name": "Name",
        "suspicion_score": "Suspicion Score"
    },
    hide_index=True
)

# Person details section
st.markdown("<div class='sub-header'>Person Details</div>", unsafe_allow_html=True)

# Select a person
selected_person_id = st.selectbox("Select a person to view details:", 
                                 options=people_df['person_id'].tolist(),
                                 format_func=lambda x: f"Person {x} (Score: {people_df[people_df['person_id'] == x]['suspicion_score'].values[0]:.2f})")

# Get the selected person's data
person_data = people_df[people_df['person_id'] == selected_person_id].iloc[0]
person_content = content_df[content_df['person_id'] == selected_person_id].sort_values(by='suspicion_score', ascending=False)
person_entities = entity_df[entity_df['person_id'] == selected_person_id].iloc[0]

# Display person details
person_col1, person_col2 = st.columns([1, 2])

with person_col1:
    st.markdown(f"""
    <div class='card'>
        <p><b>Name:</b> {person_data['name']}</p>
        <p><b>Suspicion Score:</b> {format_suspicion(person_data['suspicion_score'])}</p>
        <p><b>Group Membership:</b> {person_data['group_membership']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display entity extraction for this person
    with st.container():
        st.markdown("<b>Locations Mentioned:</b>", unsafe_allow_html=True)
        if person_entities['locations']:
            st.write(", ".join(person_entities['locations']))
        else:
            st.write("None")
        
        st.markdown("<b>Species Mentioned:</b>", unsafe_allow_html=True)
        if person_entities['species']:
            st.write(", ".join(person_entities['species']))
        else:
            st.write("None")
        
        st.markdown("<b>People Mentioned:</b>", unsafe_allow_html=True)
        if person_entities['mentioned_people']:
            st.write(", ".join(person_entities['mentioned_people']))
        else:
            st.write("None")

# Content board
with person_col2:
    st.markdown("<b>Content Board (Ranked by Suspicion Score)</b>", unsafe_allow_html=True)
    
    if len(person_content) == 0:
        st.write("No content available for this person.")
    else:
        for _, content in person_content.iterrows():
            # Generate a fake image for content with media
            if content['has_media']:
                # Use a consistent color based on content_id for better caching
                color = (
                    (content['content_id'] * 73) % 255,
                    (content['content_id'] * 31) % 255,
                    (content['content_id'] * 47) % 255
                )
                # Create a colored rectangle as a placeholder image
                img = Image.new('RGB', (300, 200), color=color)
                
                # Add a watermark with content ID for better visual identification
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                
                fig = Figure(figsize=(3, 2), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Content #{content['content_id']}", 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, color='white')
                ax.axis('off')
                fig.patch.set_facecolor(tuple(c/255 for c in color))
                
                # Convert to image
                img_byte_arr = io.BytesIO()
                fig.savefig(img_byte_arr, format='PNG', bbox_inches='tight', pad_inches=0)
                img_byte_arr.seek(0)
                img_data = img_byte_arr.getvalue()
                
                media_html = f"""
                <img src="data:image/png;base64,{base64.b64encode(img_data).decode()}" 
                     style="width:100%; max-width:300px; border-radius:5px; margin-top:10px;">
                """
            else:
                media_html = ""
            
            # Format the content card
            st.markdown(f"""
            <div class='card'>
                <p><b>{content['content_type'].title()}</b> - 
                   <span>Suspicion Score: {format_suspicion(content['suspicion_score'])}</span> - 
                   <span>{content['timestamp'].strftime('%Y-%m-%d %H:%M')}</span></p>
                <p>{content['content_text']}</p>
                {media_html}
            </div>
            """, unsafe_allow_html=True)

# Add a footer with information
st.markdown("---")
st.markdown("""
<footer>
    <p>Dashboard created for Electric Twins Hackathon project.</p>
    <p>Data is simulated for demonstration purposes only.</p>
    <p>© 2025 Social Media Analysis Tool</p>
</footer>
""", unsafe_allow_html=True)