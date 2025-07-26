# app.py - Streamlit Dashboard for Global Aviation Analytics

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
import numpy as np





# Load preprocessed data
airlines = pd.read_csv("clean_airlines.csv")
airports = pd.read_csv("clean_airports.csv")
routes = pd.read_csv("routes_with_distance.csv")
centrality = pd.read_csv("network_centrality_metrics.csv")
airline_market = pd.read_csv("airline_market_share.csv")

# Additional business intelligence data
competitive_routes = pd.read_csv("competitive_routes.csv")
underserved = pd.read_csv("underserved_city_pairs.csv")

st.set_page_config(page_title="Global Aviation Network Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title('Global Aviation Analytics Dashboard')
page = st.sidebar.radio("Navigate to:", ["Overview", "Network", "Geography", "Business", "Search"])

# Overview Tab
if page == "Overview":
    st.title("Global Aviation Overview")
    col1, col2, col3 = st.columns(3)
    

    
    col1.metric("Total Airports", len(airports))
    col2.metric("Total Routes", len(routes))
    col3.metric("Top Hub", centrality.iloc[0]['Airport'])

    col4, col5, col6 = st.columns(3)
    # Top country by number of airports
    top_country = airports['country'].value_counts().idxmax()
    top_country_count = airports['country'].value_counts().max()
    col4.metric("Top Country by Airports", f"{top_country} ({top_country_count})")

    # Airline with most routes
    top_airline = routes['airline_name'].value_counts().idxmax()
    col5.metric("Airline with Most Routes", top_airline)

    # Most common route type (Short/Medium/Long)
    if 'distance_category' in routes.columns:
                                most_common_category = routes['distance_category'].mode()[0]
                                col6.metric("Most Common Route Type", most_common_category)
    else:
        col6.metric("Most Common Route Type", "N/A")   


    st.subheader("Route Distance Distribution")
    if 'distance_km' in routes.columns:
        bins = [0, 1500, 3500, float('inf')]
        labels = ['Domestic', 'Both', 'International']
        routes['distance_category'] = pd.cut(routes['distance_km'], bins=bins, labels=labels)

        distance_counts = routes['distance_category'].value_counts().sort_index()
        fig, ax = plt.subplots()
        distance_counts.plot(kind='bar', color='green', ax=ax)
        ax.set_title("Route Distance Categories")
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Routes")
        st.pyplot(fig)
    else:
        st.warning("Distance data not available to categorize routes.")
    
    st.subheader("Top 100 Airports by Connectivity (Map Preview)")

    # Top 100 busiest airports (by connections)
    connections = pd.concat([routes['source_airport'], routes['dest_airport']]).value_counts().head(100)
    top_airports = airports[airports['iata'].isin(connections.index)]

    fig_map = px.scatter_geo(top_airports, lat='latitude', lon='longitude', 
                            hover_name='name', color='country',
                            title='Top 100 Connected Airports',
                            projection='natural earth')
    fig_map.update_layout(showlegend=False)
    st.plotly_chart(fig_map, use_container_width=True)

    

    
# Network Tab
elif page == "Network":
    st.title("Network Analysis")
    
    st.subheader("Top 10 Hub Airports by Degree Centrality")
    st.dataframe(centrality[['Airport', 'Degree Centrality']].head(10))

    st.subheader("Network Graph - Top 50 Routes")
    subset_routes = routes.head(50)
    G_subset = nx.from_pandas_edgelist(subset_routes, 'source_airport', 'dest_airport', create_using=nx.DiGraph())
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G_subset, with_labels=True, node_size=500, node_color='lightblue', font_size=8, arrows=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Comprehensive Network Metrics")
    G_full = nx.from_pandas_edgelist(routes, 'source_airport', 'dest_airport', create_using=nx.DiGraph())

    # Centrality Measures
    degree_centrality = nx.degree_centrality(G_full)
    betweenness_centrality = nx.betweenness_centrality(G_full)
    closeness_centrality = nx.closeness_centrality(G_full)
    pagerank = nx.pagerank(G_full)

    def top_n(metric_dict, title):
        sorted_metric = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write(f"**Top 5 Airports by {title}:**")
        for code, value in sorted_metric:
            name = airports.loc[airports['iata'] == code, 'name'].values
            st.write(f"{code} ({name[0] if len(name) > 0 else 'Unknown'}): {value:.4f}")

    top_n(degree_centrality, "Degree Centrality")
    top_n(betweenness_centrality, "Betweenness Centrality")
    top_n(closeness_centrality, "Closeness Centrality")
    top_n(pagerank, "PageRank")

    # Network Diameter and Average Path Length
    # Use strongly connected component to avoid infinite path errors
    if nx.is_strongly_connected(G_full):
        diameter = nx.diameter(G_full)
        avg_path_len = nx.average_shortest_path_length(G_full)
    else:
        # Find the largest strongly connected component (SCC)
        largest_scc = max(nx.strongly_connected_components(G_full), key=len)
        subgraph = G_full.subgraph(largest_scc)
        diameter = nx.diameter(subgraph)
        avg_path_len = nx.average_shortest_path_length(subgraph)

    st.write(f"**Network Diameter:** {diameter}")
    st.write(f"**Average Shortest Path Length:** {avg_path_len:.2f}")


    

    # Clustering Coefficient
    clustering = nx.average_clustering(G_full.to_undirected())
    st.write(f"**Average Clustering Coefficient:** {clustering:.4f}")

    # Community Detection
    communities = greedy_modularity_communities(G_full.to_undirected())
    st.write(f"**Detected Communities:** {len(communities)}")

    # Path Analysis - Shortest path between two major hubs
    st.subheader("Path Analysis Between Major Hubs")
    hub1 = st.selectbox("Select Hub 1", options=airports['iata'].unique())
    hub2 = st.selectbox("Select Hub 2", options=airports['iata'].unique())

    if hub1 != hub2:
        try:
            path = nx.shortest_path(G_full, source=hub1, target=hub2)
            st.write(f"Shortest path from {hub1} to {hub2}: {path}")
            st.write(f"Path Length: {len(path) - 1}")
        except nx.NetworkXNoPath:
            st.warning(f"No path exists between {hub1} and {hub2}.")

    # Network Resilience - Simulate removal of top hub
    st.subheader("Network Resilience Simulation")
    top_hub = centrality.iloc[0]['Airport']
    G_resilient = G_full.copy()
    if G_resilient.has_node(top_hub):
        G_resilient.remove_node(top_hub)
        st.write(f"Removed top hub: {top_hub}")
        new_components = nx.number_weakly_connected_components(G_resilient)
        st.write(f"New number of weakly connected components: {new_components}")

    # Identify isolated nodes
    isolated_nodes = list(nx.isolates(G_full))
    st.write(f"**Number of Isolated Airports:** {len(isolated_nodes)}")

# Geography Tab
elif page == "Geography":
    st.title("Geographic Distribution of Airports")

    st.subheader("Top 10 Countries by Number of Airports")

    # Count airports per country
    country_airport_counts = airports['country'].value_counts().head(10).reset_index()
    country_airport_counts.columns = ['country', 'airport_count']

    # Bar chart
    fig_country, ax_country = plt.subplots()
    country_airport_counts.plot(kind='barh', x='country', y='airport_count', color='blue', ax=ax_country)
    ax_country.set_title("Top 10 Countries by Number of Airports")
    ax_country.set_xlabel("Number of Airports")
    ax_country.set_ylabel("Countries")
    st.pyplot(fig_country)

    
    # Scatter map of all airports
    fig = px.scatter_geo(airports, lat='latitude', lon='longitude', hover_name='name', 
                         color='country', title='Global Airport Locations', 
                         projection='natural earth')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap of Airport Density
    st.subheader("Global Airport Density Heatmap")
    heat_data = [[row['latitude'], row['longitude']] for _, row in airports.iterrows()]
    heat_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data).add_to(heat_map)
    st.components.v1.html(heat_map._repr_html_(), height=500)

    # Geographic Clustering of Airports
    st.subheader("Geographic Clustering of Airports")
    coords = airports[['latitude', 'longitude']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = 500 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    airports['cluster'] = db.labels_

    fig_cluster = px.scatter_geo(airports, lat='latitude', lon='longitude', color='cluster', 
                                hover_name='name', title='Airport Clusters', 
                                projection='natural earth')
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Airline-Specific Route Map
    st.subheader("Airline-Specific Route Map")
    airline_names = routes['airline_name'].dropna().unique()
    selected_airline = st.selectbox("Select an Airline", options=airline_names)
    airline_routes = routes[routes['airline_name'] == selected_airline]

    fig_routes = px.scatter_geo()
    for _, row in airline_routes.iterrows():
        fig_routes.add_trace(px.line_geo(lat=[row['source_lat'], row['dest_lat']],
                                     lon=[row['source_lon'], row['dest_lon']]).data[0])

    fig_routes.update_layout(title=f"Routes Operated by {selected_airline}", showlegend=False,
                         geo=dict(projection_type='natural earth'))
    st.plotly_chart(fig_routes, use_container_width=True)

# Business Tab
elif page == "Business":
    st.title("Business Intelligence - Market Analysis")
    
    st.subheader("Airline Activity Status (Active vs Inactive)")

    # Count active/inactive airlines
    activity_counts = airlines['active'].value_counts().reset_index()
    activity_counts.columns = ['status', 'count']

    # Define color map
    color_map = {'Y':'green', 'N':'red', 'n':'gray'}

    # Pie chart
    fig_activity = px.pie(activity_counts, names='status', values='count', title='Airline Operational Status', color='status', color_discrete_map=color_map)
    st.plotly_chart(fig_activity)

    st.subheader("Top 10 Airlines by Route Count")
    airline_market_sorted = airline_market.sort_values(by='route_count', ascending=False).head(10)
    st.bar_chart(airline_market_sorted.set_index('airline_name')['route_count'])

    st.subheader("Market Share by Region")
    region_share = airline_market.groupby('region')['route_count'].sum().reset_index()
    fig_region = px.pie(region_share, names='region', values='route_count', title='Airline Market Share by Region')
    st.plotly_chart(fig_region)

    st.subheader("Competition on Popular Routes")
    st.dataframe(competitive_routes.head(10))

    st.subheader("Underserved City Pairs")
    st.dataframe(underserved.head(10))

    st.subheader("Airlines with Longest Average Route Distances")
    # Calculate average distance per airline
    avg_distance_df = routes.groupby('airline_name')['distance_km'].mean().reset_index()
    avg_distance_df.columns = ['airline_name', 'avg_distance_km']

    # Top 10 airlines with longest routes
    longest_routes = avg_distance_df.sort_values(by='avg_distance_km', ascending=False).head(10)

    # Plot bar chart
    fig_long, ax_long = plt.subplots()
    longest_routes.plot(kind='barh', x='airline_name', y='avg_distance_km', color='purple', ax=ax_long)
    ax_long.set_title("Top 10 Airlines by Average Route Distance")
    ax_long.set_xlabel("Average Distance (km)")
    ax_long.set_ylabel("Airline Name")
    st.pyplot(fig_long)

    st.subheader("Airline Network Types Comparison")
    selected_airlines = st.multiselect("Select Airlines for Comparison", options=airline_market['airline_name'].unique())
    for airline in selected_airlines:
        st.write(f"**{airline} Network Summary**")
        airline_data = airline_market[airline_market['airline_name'] == airline]
        st.write(airline_data)

# Search Tab
elif page == "Search":
    st.title("Route Finder and Airline Comparison")
    
    source_airport = st.selectbox("Select Source Airport", options=routes['source_airport'].unique())
    filtered_routes = routes[routes['source_airport'] == source_airport]
    st.write(filtered_routes[['source_airport', 'dest_airport', 'airline_name', 'distance_km']].head(10))

    st.subheader("Compare Airlines by Route Count")
    airline_market_sorted = airline_market.sort_values(by='route_count', ascending=False).head(10)
    fig, ax = plt.subplots()
    airline_market_sorted.plot(kind='barh', x='airline_name', y='route_count', color='orange', ax=ax)
    ax.set_title('Top Airlines by Routes')
    ax.set_xlabel('Number of Routes')
    ax.set_ylabel('Airline Name')
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed July 2025")