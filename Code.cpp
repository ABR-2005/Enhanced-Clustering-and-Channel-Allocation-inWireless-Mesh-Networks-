#include <bits/stdc++.h>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <ctime>   // For time()
#include <cstdlib> // For rand()
#include <random>  // For std::mt19937 and std::uniform_real_distribution
#include <chrono>
using namespace std;

const int MAX = 400;              // Define the size of the graph
const int TRANSMISSION_RANGE = 5; // Transmission range for the network

struct Cluster
{
    vector<int> nodes;
    int clusterHead = -1;             // Initialize clusterHead with -1 to indicate uninitialized value
    int channel = -1;                 // Channel assigned to the cluster
    vector<int> interClusterChannels; // Channels assigned to inter-cluster links
    vector<int> neighbour;
};

// Function to calculate the distance between two nodes
int distance(int x1, int y1, int x2, int y2)
{
    return sqrt(pow(abs(x1 - x2), 2) + pow(abs(y1 - y2), 2));
}
// vector<vector<int>> generate_random_graph() {
//     // Seed the random number generator
//     srand(time(NULL));

//     vector<vector<int>> graph(MAX, vector<int>(MAX, 0));

//     // Fill the graph with random edges
//     for (int i = 0; i < MAX; ++i) {
//         for (int j = i; j < MAX; ++j) { // Since the graph is undirected, we only need to fill half of it
//             if (i == j) {
//                 graph[i][j] = 0; // No self-loops
//             } else {
//                 // Randomly assign 0 or 1 based on probabilities
//                 int random_value = rand() % 100; // Generate a random number between 0 and 99
//                 if (random_value < 80) { // 80% chance for 0
//                     graph[i][j] = graph[j][i] = 0;
//                 } else { // 20% chance for 1
//                     graph[i][j] = graph[j][i] = 1;
//                 }
//             }
//         }
//     }

//     cout << "Generated Graph:" << endl;
//     for (int i = 0; i < MAX; ++i) {
//         for (int j = 0; j < MAX; ++j) {
//             cout << graph[i][j] << " ";
//         }
//         cout << endl;
//     }

//     return graph;
// }
// Function to generate a random mesh network graph with transmission range
vector<vector<int>> generate_random_graph()
{
    //     vector<vector<int>> graph={
    //         {0, 1, 1, 0, 0, 1, 1, 0, 0, 0,},
    //         {1, 0, 0, 1, 0, 0, 1, 0, 0, 0,},
    //         {1, 0, 0, 1, 0, 0, 0, 0, 0, 1,},
    //         {0, 1, 1, 0, 1, 0, 0, 1, 0, 0,},
    //         {0, 0, 0, 1, 0, 1, 0, 1, 1, 0,},
    //         {1, 0, 0, 0, 1, 0, 0, 0, 1, 1,},
    //         {1, 1, 0, 0, 0, 0, 0, 0, 0, 1,},
    //         {0, 0, 0, 1, 1, 0, 0, 0, 1, 0,},
    //         {0, 0, 0, 0, 1, 1, 0, 1, 0, 0,},
    //         {0, 0, 1, 0, 0, 1, 1, 0, 0, 0,}

    // };

    srand(time(nullptr)); // Seed random number generator with current time
    vector<vector<int>> graph(MAX, vector<int>(MAX, 0));
    // Generate random coordinates for each node
    vector<pair<int, int>> coordinates(MAX);
    for (int i = 0; i < MAX; ++i)
    {
        coordinates[i] = {rand() % 100, rand() % 100}; // Random coordinates in a 100x100 grid
    }

    // Fill the graph based on transmission range
    for (int i = 0; i < MAX; ++i)
    {
        for (int j = i + 1; j < MAX; ++j)
        {
            int dist = distance(coordinates[i].first, coordinates[i].second, coordinates[j].first, coordinates[j].second);
            if (dist <= TRANSMISSION_RANGE)
            {
                graph[i][j] = graph[j][i] = 1; // Nodes are within transmission range
            }
        }
    }
    for (int i = 0; i < MAX; ++i)
    {
        graph[i][i] = 0;
    }

    cout << "Generated Graph:" << endl;
    for (int i = 0; i < MAX; ++i)
    {
        for (int j = 0; j < MAX; ++j)
        {
            cout << graph[i][j] << " ";
        }
        cout << endl;
    }
    return graph;
}

// Function to find maximal cliques using Bron-Kerbosch algorithm
void find_maximal_cliques(
    vector<int> &R,
    vector<int> &P,
    vector<int> &X,
    vector<vector<int>> &maximalCliques,
    const vector<vector<int>> &graph)
{

    if (P.empty())
    {
        sort(R.begin(), R.end());
        maximalCliques.push_back(R);
        return;
    }

    for (int &v : P)
    {
        vector<int> newR = R;
        vector<int> newP;
        vector<int> newX;
        newR.push_back(v);
        for (int neighbor : P)
        {
            if (graph[v][neighbor])
                newP.push_back(neighbor);
        }
        for (int neighbor : X)
        {
            if (graph[v][neighbor])
                newX.push_back(neighbor);
        }
        find_maximal_cliques(newR, newP, newX, maximalCliques, graph);
        P.erase(remove(P.begin(), P.end(), v), P.end());
        X.push_back(v);
    }
}

// Function to find all maximal cliques in the graph
vector<vector<int>> find_maximal_cliques(const vector<vector<int>> &graph)
{
    vector<int> R, P, X;
    for (int i = 0; i < MAX; ++i)
        P.push_back(i);
    vector<vector<int>> maximalCliques;
    find_maximal_cliques(R, P, X, maximalCliques, graph);
    return maximalCliques;
}
// Function to find clusters from the interference graph
// Function to find clusters from the interference graph

bool cmp(vector<int> &a, vector<int> &b)
{
    return a.size() > b.size();
}
void remove(vector<vector<int>> &maximal)
{
    sort(maximal.begin(), maximal.end(), cmp);
    int a = 0;
    while (a < maximal.size())
    {
        unordered_set<int> s(maximal[a].begin(), maximal[a].end());
        for (int i = a + 1; i < maximal.size(); i++)
        {
            for (int j = 0; j < maximal[i].size(); j++)
            {
                if (s.count(maximal[i][j]))
                {
                    maximal.erase(maximal.begin() + i);
                    i--;
                    break;
                }
            }
        }
        a++;
    }
}
bool is_subset(vector<int> arr1, vector<int> arr2)
{
    int m = arr1.size();

    int n = arr2.size();

    unordered_set<int> s;

    for (int i = 0; i < m; i++)
    {
        s.insert(arr1[i]);
    }
    int p = s.size();
    for (int i = 0; i < n; i++)
    {
        s.insert(arr2[i]);
    }
    if (s.size() == p)
    {
        return true;
    }
    else
    {
        return false;
    }
}
void twoHopNeighbors(vector<vector<int>> &maximalCliques, vector<vector<int>> &graph)
{
    vector<vector<int>> ans;

    for (int i = 0; i < MAX; i++)
    {
        unordered_set<int> temp;
        for (int j = 0; j < MAX; j++)
        {
            if (!graph[i][j])
                continue;
            for (int k = 0; k < MAX; k++)
            {
                if (graph[j][k] and i != k and graph[i][k] != 1)
                    temp.insert(k);
            }
        }
        vector<int> temp1;
        for (auto x : temp)
        {
            temp1.push_back(x);
        }
        sort(temp1.begin(), temp1.end());
        for (int j = 0; j < maximalCliques.size(); j++)
        {
            if (is_subset(temp1, maximalCliques[j]))
            {
                ans.push_back(maximalCliques[j]);
                break;
            }
        }
    }
    maximalCliques = ans;
}
vector<int> removenodes(vector<vector<int>> &maximalCliques, vector<vector<int>> &graph)
{
    // Create a set to store nodes present in maximal cliques
    unordered_set<int> nodesInMaximalCliques;
    // Populate the set with nodes present in maximal cliques
    for (const auto &clique : maximalCliques)
    {
        for (int node : clique)
        {
            nodesInMaximalCliques.insert(node);
        }
    }
    // Create a vector to store nodes present in graph but not in any maximal clique
    vector<int> nodesNotInMaximalCliques;
    // Traverse through the graph and add nodes that are not in maximal cliques to the result vector
    for (int i = 0; i < graph.size(); ++i)
    {
        if (nodesInMaximalCliques.find(i) == nodesInMaximalCliques.end())
        {
            nodesNotInMaximalCliques.push_back(i);
        }
    }
    return nodesNotInMaximalCliques;
}
bool hasNeighborInCluster(int node, const vector<int> &cluster, const vector<vector<int>> &graph)
{
    for (int neighbor : cluster)
    {
        if (graph[node][neighbor] || graph[neighbor][node])
        {
            return true;
        }
    }
    return false;
}
void find_min_cluster(int not_clique, const vector<vector<int>> &graph, vector<Cluster> &clusters)
{
    vector<int> neighborClusters; // Store indices of clusters containing neighbors of not_clique
    // Find clusters containing neighbors of not_clique
    for (size_t i = 0; i < clusters.size(); ++i)
    {
        if (hasNeighborInCluster(not_clique, clusters[i].nodes, graph))
        {
            neighborClusters.push_back(i);
        }
    }
    // If no clusters contain neighbors of not_clique, return
    if (neighborClusters.empty())
    {
        // cout<<"hi";
        return;
    }
    // Find the cluster with the minimum number of nodes among clusters containing neighbors of not_clique
    auto minClusterIndex = neighborClusters[0];
    for (size_t i = 1; i < neighborClusters.size(); ++i)
    {
        if (clusters[neighborClusters[i]].nodes.size() < clusters[minClusterIndex].nodes.size())
        {
            minClusterIndex = neighborClusters[i];
        }
    }
    // Add not_clique to the minimum cluster
    clusters[minClusterIndex].nodes.push_back(not_clique);
}
void form_clusters(std::vector<Cluster> &clusters, const std::vector<std::vector<int>> &graph)
{
    for (size_t i = 0; i < clusters.size(); ++i)
    {
        if (clusters[i].nodes.size() <= 2)
        {

            // Find the cluster with the minimum number of nodes among clusters containing neighbors of cluster[i]
            std::vector<int> neighborClusters;
            for (size_t j = 0; j < clusters.size(); ++j)
            {
                if (j != i)
                {
                    bool hasNeighbor = false;
                    for (int node : clusters[i].nodes)
                    {
                        if (hasNeighborInCluster(node, clusters[j].nodes, graph))
                        {
                            hasNeighbor = true;
                            break;
                        }
                    }
                    if (hasNeighbor)
                    {
                        neighborClusters.push_back(j);
                    }
                }
            }
            int minClusterIndex = -1;
            size_t minClusterSize = std::numeric_limits<size_t>::max();
            for (int neighborIndex : neighborClusters)
            {
                if (clusters[neighborIndex].nodes.size() < minClusterSize)
                {
                    minClusterSize = clusters[neighborIndex].nodes.size();
                    minClusterIndex = neighborIndex;
                }
            }

            // Merge nodes from the current cluster to the cluster with the minimum size
            if (minClusterIndex != -1)
            {
                for (int node : clusters[i].nodes)
                {
                    clusters[minClusterIndex].nodes.push_back(node);
                }
                // Remove the current cluster
                clusters.erase(clusters.begin() + i);
                --i; // Adjust index due to cluster deletion
            }
        }
    }
}

void remove_single(vector<vector<int>> &maximalCliques)
{
    vector<int> indicesToRemove;
    for (int i = 0; i < maximalCliques.size(); ++i)
    {
        if (maximalCliques[i].size() == 1)
        {
            indicesToRemove.push_back(i);
        }
    }
    // Erase elements in reverse order to avoid invalidating indices
    for (int i = indicesToRemove.size() - 1; i >= 0; --i)
    {
        maximalCliques.erase(maximalCliques.begin() + indicesToRemove[i]);
    }
}

vector<Cluster> find_clusters(const vector<vector<int>> &graph)
{
    vector<Cluster> clusters;
    // vector<bool> nodeUsed(MAX, false);
    vector<vector<int>> graphCopy = graph; // Make a copy of the graph to modify
    // Find maximal cliques before iterating over nodes
    vector<vector<int>> maximalCliques = find_maximal_cliques(graph);
    // for(int i=0;i<graph.size();i++){
    //     for(int j=i+1;j<graph.size();j++){
    //         if(graph[i][j]){

    //             maximalCliques.push_back({i,j});
    //         }
    //     }
    // }
    remove(maximalCliques);
    remove_single(maximalCliques);
    Cluster x;
    for (int i = 0; i < maximalCliques.size(); i++)
    {
        x.nodes = maximalCliques[i];
        clusters.push_back(x);
        //  for(int j=0;j<maximalCliques[i].size();j++){
        //      cout<<maximalCliques[i][j]<<"\t";
        //  }
        // cout<<endl;
    }

    vector<int> not_clique = removenodes(maximalCliques, graphCopy);

    for (int i = 0; i < not_clique.size(); i++)
    {
        find_min_cluster(not_clique[i], graph, clusters);
    }
    form_clusters(clusters, graphCopy);
    return clusters;
}

vector<int> find_two_hop(vector<vector<int>> &graph, int k)
{
    vector<int> ans;
    // for(int i=0;i<ans.size();i++){
    //     cout<<ans[i]<<" ";
    // }
    // cout<<endl;
    unordered_set<int> temp;
    for (int i = 0; i < MAX; i++)
    {
        for (int j = 0; j < MAX; j++)
        {
            if (!graph[i][j])
                continue;

            if (graph[i][j] and k != j and graph[j][k] != 1)
                temp.insert(j);
        }
    }

    for (auto elem : temp)
    {
        // cout<<elem<<" ";
        ans.push_back(elem);
    }
    return ans;
}
void find_min_cluster2(vector<int> &not_clique, const vector<vector<int>> &graph, vector<Cluster> &clusters)
{
    vector<int> neighborClusters; // Store indices of clusters containing neighbors of not_clique
    // Find clusters containing neighbors of not_clique
    for (int j = 0; j < not_clique.size(); j++)
    {
        for (size_t i = 0; i < clusters.size(); ++i)
        {
            if (hasNeighborInCluster(not_clique[j], clusters[i].nodes, graph))
            {
                neighborClusters.push_back(i);
            }
        }
    }
    // If no clusters contain neighbors of not_clique, return
    if (neighborClusters.empty())
    {
        // cout<<"hi";
        return;
    }
    // Find the cluster with the minimum number of nodes among clusters containing neighbors of not_clique
    auto minClusterIndex = neighborClusters[0];
    for (size_t i = 1; i < neighborClusters.size(); ++i)
    {
        if (clusters[neighborClusters[i]].nodes.size() < clusters[minClusterIndex].nodes.size())
        {
            minClusterIndex = neighborClusters[i];
        }
    }
    // Add not_clique to the minimum cluster
    for (int i = 0; i < not_clique[i]; i++)
        clusters[minClusterIndex].nodes.push_back(not_clique[i]);
}
vector<Cluster> find_clusters_two(const vector<vector<int>> &graph)
{
    vector<Cluster> clusters;
    // vector<bool> nodeUsed(MAX, false);
    vector<vector<int>> graphCopy = graph; // Make a copy of the graph to modify
    // Find maximal cliques before iterating over nodes
    vector<vector<int>> maximalCliques = find_maximal_cliques(graph);
    for (int i = 0; i < graph.size(); i++)
    {
        for (int j = i + 1; j < graph.size(); j++)
        {
            if (graph[i][j])
            {

                maximalCliques.push_back({i, j});
            }
        }
    }
    remove_single(maximalCliques);

    twoHopNeighbors(maximalCliques, graphCopy);

    remove(maximalCliques);
    Cluster x;
    for (int i = 0; i < maximalCliques.size(); i++)
    {
        x.nodes = maximalCliques[i];
        clusters.push_back(x);
    }

    vector<int> not_clique = removenodes(maximalCliques, graphCopy);

    for (int i = 0; i < graphCopy.size(); i++)
    {
        for (int k = 0; k < not_clique.size(); k++)
            if (i == not_clique[k])
                continue;

        for (int j = 0; j < graphCopy.size(); j++)
        {
            graphCopy[i][j] = 0;
            graphCopy[j][i] = 0;
        }
    }
    // cout<<endl;
    maximalCliques = find_maximal_cliques(graphCopy);
    remove_single(maximalCliques);
    sort(maximalCliques.begin(), maximalCliques.end(), cmp);
    // cout<<endl;
    for (int i = 0; i < not_clique.size(); i++)
    {
        vector<int> y = find_two_hop(graphCopy, not_clique[i]);
        if (y.size() == 0)
        {
            // cout<<endl;
            find_min_cluster(not_clique[i], graph, clusters);
        }
        else
        {
            for (int j = 0; j < maximalCliques.size(); j++)
            {
                if (is_subset(y, maximalCliques[j]))
                {
                    if (maximalCliques[j].size() <= 2)
                    {
                        find_min_cluster2(maximalCliques[j], graph, clusters);
                    }
                    else
                    {
                        x.nodes = maximalCliques[i];
                        clusters.push_back(x);
                    }
                }
            }
        }
    }
    return clusters;
}
void calculate_and_display_edges(const vector<Cluster> &clusters, const vector<vector<int>> &graph)
{
    // Calculate and display edges within clusters for each clustering algorithm
    // cout << "Edges within clusters:" << endl;
    int totalEdges = 0;
    for (int i = 0; i < clusters.size(); ++i)
    {
        int edgesWithinCluster = 0;
        for (int j = 0; j < clusters[i].nodes.size(); ++j)
        {
            for (int k = j + 1; k < clusters[i].nodes.size(); ++k)
            {
                if (graph[clusters[i].nodes[j]][clusters[i].nodes[k]])
                {
                    ++edgesWithinCluster;
                }
            }
        }
        totalEdges += edgesWithinCluster;
        // cout << "Cluster " << i + 1 << ": " << edgesWithinCluster << " edges" << endl;
    }
    cout << "Total edges within clusters: " << totalEdges << endl;
}
double calculate_modularity(const Cluster &cluster, const vector<vector<int>> &graph)
{
    // Calculate the total number of intra-cluster edges
    int totalIntraClusterEdges = 0;
    for (int i = 0; i < cluster.nodes.size(); ++i)
    {
        for (int j = i + 1; j < cluster.nodes.size(); ++j)
        {
            if (graph[cluster.nodes[i]][cluster.nodes[j]])
            {
                ++totalIntraClusterEdges;
            }
        }
    }

    // Calculate the total number of nodes within the cluster
    int totalNodesInCluster = cluster.nodes.size();

    // Calculate the total number of edges in the network (m)
    int totalEdgesInNetwork = 0;
    for (int i = 0; i < graph.size(); ++i)
    {
        for (int j = i + 1; j < graph.size(); ++j)
        {
            if (graph[i][j])
            {
                ++totalEdgesInNetwork;
            }
        }
    }

    // Calculate modularity for the cluster
    double modularity = (double)totalIntraClusterEdges / totalEdgesInNetwork - pow((double)(2 * totalIntraClusterEdges) / (2 * totalEdgesInNetwork), 2);

    return modularity;
}

void calculate_total_modularity(const vector<Cluster> &clusters, const vector<vector<int>> &graph)
{
    double totalModularity = 0.0;
    for (const Cluster &cluster : clusters)
    {
        double clusterModularity = calculate_modularity(cluster, graph);
        totalModularity += clusterModularity;
    }
    cout << totalModularity << endl;
}

double calculateMean(const std::vector<Cluster> &clusters)
{
    double sum = 0.0;
    for (const auto &cluster : clusters)
    {
        sum += cluster.nodes.size();
    }
    return sum / clusters.size();
}

// Function to calculate the standard deviation of cluster sizes
double calculateStandardDeviation(const std::vector<Cluster> &clusters, double mean)
{
    double variance = 0.0;
    for (const auto &cluster : clusters)
    {
        variance += pow(cluster.nodes.size() - mean, 2);
    }
    variance /= clusters.size();
    return sqrt(variance);
}

// Function to calculate the coefficient of variance for cluster sizes
double calculateCoefficientOfVariance(const std::vector<Cluster> &clusters)
{
    double mean = calculateMean(clusters);
    double standardDeviation = calculateStandardDeviation(clusters, mean);
    return (100.0 * standardDeviation) / mean;
}
void calculate_avg_cluster_size(const vector<Cluster> &clusters)
{
    double totalNodes = 0;
    for (const auto &cluster : clusters)
    {
        totalNodes += cluster.nodes.size();
    }
    cout << totalNodes / clusters.size() << endl;
}
int main()
{
    // Generate random graph representing the mesh network
    vector<vector<int>> graph = generate_random_graph();
    // Find clusters in the network
    cout << "our method" << endl;
    auto start_time = chrono::steady_clock::now();
    vector<Cluster> clusters = find_clusters(graph);

    auto end_time = chrono::steady_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); // Calculate elapsed time
    cout << "Time taken by our method: " << elapsed_time.count() << " milliseconds" << endl;
    calculate_and_display_edges(clusters, graph);
    cout << "CCCA" << endl;
    start_time = chrono::steady_clock::now();
    vector<Cluster> clusters2 = find_clusters_two(graph);

    end_time = chrono::steady_clock::now();
    elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); // Calculate elapsed time
    cout << "Time taken by CCCA: " << elapsed_time.count() << " milliseconds" << endl;
    calculate_and_display_edges(clusters2, graph);
    cout << "MODULARITY\n";
    cout << "our method:";
    calculate_total_modularity(clusters, graph);
    cout << "CCCA:";
    calculate_total_modularity(clusters2, graph);
    cout << "COEFFICIENT OF VARIANCE\n";
    cout << "our method\n";
    double coefficientOfVariance = calculateCoefficientOfVariance(clusters);
    cout << "Coefficient of Variance: " << coefficientOfVariance << std::endl;
    cout << "CCCa";
    coefficientOfVariance = calculateCoefficientOfVariance(clusters2);
    cout << "Coefficient of Variance: " << coefficientOfVariance << std::endl;
    cout << "avg no of nodes\n";
    cout << "our method:";
    calculate_avg_cluster_size(clusters);
    cout << "CCCA:";
    calculate_avg_cluster_size(clusters2);
    // cout<<"our method"<<endl;
    //  for(int i=0;i<clusters.size();i++){
    //     for(int j=0;j<clusters[i].nodes.size();j++){
    //         cout<<clusters[i].nodes[j]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"CCCA"<<endl;
    // for(int i=0;i<clusters2.size();i++){
    //     for(int j=0;j<clusters2[i].nodes.size();j++){
    //         cout<<clusters2[i].nodes[j]<<" ";
    //     }
    //     cout<<endl;
    // }
}