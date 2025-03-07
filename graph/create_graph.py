import sqlite3
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import node_classification
import pickle


def extract_data_with_query(query):
    # Connect to SQLite database
    conn = sqlite3.connect("../social_network_anonymized.db")

    # Execute the query
    data = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return data

def create_person_graph_with_relationship(people_profiles, people_connections, only_connected_nodes=False,friends_conn=False, group_conn=False, follow_conn=False, comment_conn=False,tagged_conn=False):
    # From the same region
    # Follow them on FB
    # Friends with each other
    # In the same group
    # Create an empty undirected graph
    G = nx.Graph()
    # Add nodes (persons)
    people_df = people_profiles[people_profiles.profile_type=="person"]
    for person_id in people_df.id.unique():
        # TODO: add more attributes here
        G.add_node(person_id, region=people_df[people_df.id==person_id].region.values[0])
    relationships = []
    if friends_conn:
        friend_connections_1 = people_connections[people_connections["connection_type"] == "updated-friends-list-on-facebook"]
        friend_connections_1['edge_type'] = "friend_with"
        friend_edges_list_1 = friend_connections_1[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(friend_edges_list_1)} friends connections")
        relationships.extend(friend_edges_list_1)
        friend_connections_2 = people_connections[people_connections["connection_type"] == "ADDED_THEM_AS_A_FRIEND_ON_FACEBOOK"]
        friend_connections_2['edge_type'] = "friend_with"
        friend_edges_list_2 = friend_connections_2[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(friend_edges_list_2)} friends connections")
        relationships.extend(friend_edges_list_2)
    if group_conn:
        group_connections = people_connections[people_connections["connection_type"] == "BECAME_MEMBER_OF_GROUP_ON_FACEBOOK"]
        group_connections['edge_type'] = "in_same_group"
        group_edges_list = group_connections[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(group_edges_list)} group connections")
        relationships.extend(group_edges_list)
    if follow_conn:
        follow_connections = people_connections[people_connections["connection_type"] == "FOLLOWED_THEM_ON_FACEBOOK"]
        follow_connections['edge_type'] = "follower"
        follow_edges_list = follow_connections[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(follow_edges_list)} follow connections")
        relationships.extend(follow_edges_list)
    if comment_conn:
        comment_connections = people_connections[people_connections["connection_type"] == "COMMENTED_ON_THEIR_POST_ON_FACEBOOK"]
        comment_connections['edge_type'] = "commented_on"
        comment_edges_list = comment_connections[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(comment_edges_list)} comment connections")
        relationships.extend(comment_edges_list)
    if tagged_conn:
        tagged_connections = people_connections[people_connections["connection_type"] == "MENTIONED_THEM_ON_FACEBOOK"]
        tagged_connections['edge_type'] = "tagged"
        tagged_edges_list = tagged_connections[["source_id", "target_id", "edge_type", "id"]].values.tolist()
        print(f"Adding {len(tagged_edges_list)} tagged connections")
        relationships.extend(tagged_edges_list)
    print(f"Adding {len(relationships)} total connections")
    for u, v, label, id in relationships:
        G.add_edge(u, v, label=label, unique_id = id)
    if only_connected_nodes:
        # Filter only connected nodes
        print(f"Number of nodes: {G.number_of_nodes()}")
        connected_nodes = {node for edge in relationships for node in edge}  # Get unique nodes with at least one edge
        print(f"Number of connected nodes: {len(connected_nodes)}")
        G = G.subgraph(connected_nodes)  # Create subgraph with only connected nodes
    return G


def label_nodes_in_a_graph(graph_object, node_list, label):
    for node in node_list:
        graph_object.nodes[node]["label"] = label
    return graph_object


if __name__ == "__main__":
    # People list
    people_profiles = extract_data_with_query("SELECT * FROM Profiles")
    # People network
    people_connections = extract_data_with_query("SELECT * FROM ProfileConnection")
    graph = create_person_graph_with_relationship(people_profiles, people_connections, only_connected_nodes=False, friends_conn=True, group_conn=True, follow_conn=True, comment_conn=True,tagged_conn=True)
    nx.write_graphml(graph, "overall_graph.graphml")
    extra_data = pd.read_parquet("translated_posts.parquet")
    profile_actvitity_link = extract_data_with_query("SELECT * FROM ProfileActivity")
    combined_data = pd.merge(extra_data, profile_actvitity_link[["profile_id", "activity_id"]], left_on='id', right_on='activity_id', how='left')
    combined_data = combined_data.dropna(subset=['profile_id'])
    combined_data['profile_id'] = combined_data['profile_id'].astype(int)
    target_nodes = set(combined_data['profile_id'])
    neighbors = set()
    for node in target_nodes:
        neighbors.update(graph.neighbors(node))
    # Combine target nodes with their neighbors
    nodes_to_include = target_nodes.union(neighbors)
    # Create subgraph with the selected nodes
    subgraph = graph.subgraph(nodes_to_include)
    print(f"Number of nodes in subgraph: {subgraph.number_of_nodes()}")
    print(f"Number of edges in subgraph: {subgraph.number_of_edges()}")
    nx.write_graphml(subgraph, "subgraph.graphml")
    # node classification
    traffic_likelihood = combined_data.groupby("profile_id").sum()[["traffic_likelihood"]]
    example_nodes_suspicious = traffic_likelihood[traffic_likelihood['traffic_likelihood'] >= 100].index.tolist()
    example_nodes_not_suspicious = traffic_likelihood[traffic_likelihood['traffic_likelihood'] <= 1].index.tolist()
    # change them all into int
    example_nodes_suspicious = [int(node) for node in example_nodes_suspicious]
    example_nodes_not_suspicious = [int(node) for node in example_nodes_not_suspicious]
    subgraph = label_nodes_in_a_graph(subgraph, example_nodes_suspicious, "suspicious")
    subgraph = label_nodes_in_a_graph(subgraph, example_nodes_not_suspicious, "not_suspicious")
    predictions = node_classification.harmonic_function(subgraph)
    # now loop through the predictions and nodes to assign labels to node attribute
    for prediction, node in zip(predictions, subgraph.nodes()):
        subgraph.nodes[node]["graph_based_prediction"] = prediction
        if node in traffic_likelihood.index:
            subgraph.nodes[node]["llm_based_prediction"] = traffic_likelihood.loc[node]["traffic_likelihood"]
        else:
            subgraph.nodes[node]["llm_based_prediction"] = "no post data"
    nx.write_graphml(subgraph, "subgraph_with_predictions.graphml")
    with open("subgraph_with_predictions.pickle", "wb") as f:
        pickle.dump(subgraph, f)
    # pickle this graph
    # plt.figure(figsize=(12, 12))
    # pos = nx.spring_layout(subgraph, k=0.1)  # Adjust k for better spacing
    # nx.draw(graph, pos, node_size=0.5, width=0.1, edge_color="gray", with_labels=False)
    # plt.savefig("graph.png")

    

        
