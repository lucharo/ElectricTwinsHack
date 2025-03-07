import plotly.graph_objects as go
import networkx as nx


def select_subgraph_with_single_node(graph, node):
    neighbors = set()
    target_nodes = {node}
    for node in target_nodes:
        neighbors.update(graph.neighbors(node))
    # Combine target nodes with their neighbors
    nodes_to_include = target_nodes.union(neighbors)
    # Create subgraph with the selected nodes
    subgraph = graph.subgraph(nodes_to_include)
    print(f"Number of nodes in subgraph: {subgraph.number_of_nodes()}")
    print(f"Number of edges in subgraph: {subgraph.number_of_edges()}")
    return subgraph


def plot_subgraph_in_plotly(subgraph):
    # Compute positions using spring layout
    pos = nx.spring_layout(subgraph)

    # Add positions as node attributes
    for node, (x, y) in pos.items():
        subgraph.nodes[node]['pos'] = (x, y)
    
    # NODE TRACE
    node_x = []
    node_y = []
    node_hover_text = []
    node_adjacencies = []
    
    for node in subgraph.nodes():
        x, y = subgraph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        
        # Count adjacencies for node color
        adjacencies = list(subgraph.neighbors(node))
        node_adjacencies.append(len(adjacencies))
        
        # Create hover text with all node attributes
        hover_text = f"Node: {node}<br>Connections: {len(adjacencies)}<br>"
        
        # Add all node attributes to hover text
        for key, value in subgraph.nodes[node].items():
            if key != 'pos':  # Skip position data
                hover_text += f"{key}: {value}<br>"
                
        node_hover_text.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_hover_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_adjacencies,
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                text='Node Connections',
                side='right'
                ),
                xanchor='left',
            ),
            line_width=2))
    
    # EDGE TRACES - Create a trace with invisible line and visible markers along the edge
    edge_traces = []
    
    for edge in subgraph.edges():
        # Get node positions
        x0, y0 = subgraph.nodes[edge[0]]['pos']
        x1, y1 = subgraph.nodes[edge[1]]['pos']
        
        # Create line trace for the edge
        line_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Get edge attributes for hover text
        edge_attrs = subgraph.get_edge_data(edge[0], edge[1])
        hover_text = f"Edge: {edge[0]} → {edge[1]}<br>"
        
        # Add all edge attributes to hover text
        for key, value in edge_attrs.items():
            if key != 'pos':
                hover_text += f"{key}: {value}<br>"
        
        # Create marker trace for the middle of the edge (this will be hoverable)
        # Calculate the middle point of the edge
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        marker_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(
                size=5,
                color='#888',
                opacity=0.5
            ),
            hoverinfo='text',
            text=hover_text,
            showlegend=False
        )
        
        edge_traces.append(line_trace)
        edge_traces.append(marker_trace)
    
    # Combine all traces
    all_traces = edge_traces + [node_trace]
    
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    
    return fig

