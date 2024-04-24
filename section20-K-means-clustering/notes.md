# K-Means Clustering
- Is an unsupervised learning algorithm that will attempt to group similar clusters together in your data.
- The overall goal is to divide data into distinct groups such that observations within each group are similar
- The K Means Algorithm:
    - Chooses a number of K clusters
    - Randomly assign each point to a cluster
    - Until the cluster stop changing, repeat the following:
        - For each cluster, compute the cluster centroid by taking the mean vector of points in the cluster
        - Assign each data point to the cluster for which the centroid is the closest
- no easy answer for choosing best k-value
- One way is the elbow method:
    - First compute the (SSE) sum of squared error for some values of k (example: 2,4,6,8)
    - The SSE is defined as the sum of the squared distance between each member of the cluster and its centroid
    - If you plot k against the SSE, you will see that the error of clusters increases, they should be samller, so distortion is also smaller.
    - The idea of the elbow method is to choose the k at which the SSE decreases abruptly
    - This produces an "elbow effect" in the graph as you can see in the following picture
    

 