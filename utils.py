from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import chat
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from collections import Counter
from scipy.stats import mode


def classify_job_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'general job classification' to a DataFrame based on 
    keywords found in the 'title' column.

    The classification logic applies rules in a specific order of precedence
    to handle complex or overlapping job titles.

    Args:
        df: The input pandas DataFrame containing a 'title' column.

    Returns:
        The modified DataFrame with the new 'general job classification' column.
    """
    if 'title' not in df.columns:
        print("Error: DataFrame must contain a 'title' column.")
        return df

    # Convert titles to lowercase for case-insensitive matching
    titles_lower = df['title'].str.lower().fillna('')

    # Use a function to apply classification rules with precedence
    def get_classification(title: str) -> str:
        """Determines the classification based on keywords in order of importance."""

        # 1. Project Management
        # Check for Program Manager or Project Manager/Engineer first
        if 'program manager' in title or 'project manager' in title or 'project engineer' in title:
            return 'Project Management'

        # 2. Data & Analytics (Includes specialized data engineering/ML roles)
        if ('data' in title and ('engineer' in title or 'analyst' in title or 'architect' in title)) or \
           'machine learning' in title or 'analytics' in title or 'etl' in title or 'data science' in title:
            return 'Data & Analytics'

        # 3. Business Analysis
        if 'business analyst' in title:
            return 'Business Analysis'
        
        # 4. Technical Architecture (Catch-all for non-data architects)
        if 'architect' in title:
            return 'Technical Architecture'

        # 5. Software Development / Engineering (The catch-all for core tech roles)
        if 'software engineer' in title or 'developer' in title or 'devops' in title or \
           'full stack' in title or 'web developer' in title or 'frontend' in title or \
           'test engineer' in title or 'qa engineer' in title or 'computer scientist' in title or 'front end' in title:
            return 'Software Development / Engineering'

        # Default classification
        return 'Other/Unclassified'

    # Apply the classification function to the lowercase titles
    df['general job classification'] = titles_lower.apply(get_classification)
    
    return df


# Define the desired output structure using Pydantic - this creates a JSON Schema that Ollama is forced to follow
class JobRequirements(BaseModel):
    """A structured model to hold the extracted skills and requirements from a job posting."""
    skills: List[str] = Field(
        ..., 
        description="A list of specific, technical, or soft skills required. E.g., 'Python', 'Machine Learning', 'Problem-Solving'."
    )
    requirements: List[str] = Field(
        ..., 
        description="A list of formal requirements, like years of experience, educational degrees, or specific certifications. E.g., 'Bachelor's Degree in Computer Science', '5+ years of experience with Kafka', 'AWS Certified'."
    )


def extract_skills_requirements(job_post: str, model_name: str = "llama3.2") -> JobRequirements:
    """
    Sends the job posting to an Ollama model and extracts structured data.
    """
    
    # Generate the JSON schema from the Pydantic model
    schema = JobRequirements.model_json_schema()

    # The system prompt guides the model's behavior
    system_prompt = (
        "You are an expert HR data extraction bot. Your task is to accurately "
        "extract the required skills and formal requirements from the provided job posting. "
        "Do not include any commentary or additional text. "
        "The output MUST conform strictly to the provided JSON schema."
    )
    
    # The user prompt contains the data to be analyzed
    user_prompt = f"Analyze the following job posting and return the extracted data:\n\n---\n{job_post}"
    
    print(f"--- Sending request to Ollama with model: {model_name} ---")

    try:
        response = chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            # This is the key setting for structured output!
            format=schema,
            # Use temperature 0 for deterministic, reliable extraction
            options={'temperature': 0}
        )
        
        # The model's content is a JSON string conforming to the schema
        json_string = response['message']['content']
        
        # Validate and convert the JSON string back into a Pydantic object
        extracted_data = JobRequirements.model_validate_json(json_string)
        
        return extracted_data

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure Ollama server is running and the model is pulled.")
        return None
    

def append_skills_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts skills and formal requirements from each job description in the DataFrame
    and appends them as new columns.

    Args:
        df: The input DataFrame with a 'description' column.

    Returns:
        The modified DataFrame with 'skills' and 'formal_requirements' columns.
    """
    skills_list = []
    requirements_list = []

    for i, job_posting_text in enumerate(df['description']):
        print(f"Processing row {i + 1}/{len(df)}...")
        result = extract_skills_requirements(job_posting_text)
        if result:
            skills_list.append(result.skills)
            requirements_list.append(result.requirements)
        else:
            skills_list.append([])  # Append empty list if extraction fails
            requirements_list.append([])

    # Add the extracted data as new columns
    df['skills'] = skills_list
    df['formal_requirements'] = requirements_list

    return df



def find_elbow_point(k_values, inertia_values):
    """
    Identifies the elbow point in the inertia curve using the maximum second derivative method.

    Args:
        k_values: List of K values.
        inertia_values: Corresponding inertia values for each K.

    Returns:
        int or None: The K value at the elbow point.
    """
    # Calculate the second derivative of inertia values
    second_derivative = np.diff(inertia_values, n=2)

    # Find peaks in the second derivative (elbow points)
    peaks, _ = find_peaks(-second_derivative)  # Negative because we want maxima in the original curve

    if len(peaks) > 0:
        elbow_k = k_values[peaks[0] + 1]  # +1 to account for the diff reducing the array size
        return elbow_k
    else:
        return None
    

def plot_elbow_curve(vectors, k_range=range(2, 25)):
    """
    Plots the Elbow Curve for K-Means clustering and identifies the optimal number of clusters.

    Args:
        skill_vectors (ndarray): The input data to cluster.
        k_range (range): The range of K values to test.

    Returns:
        int or None: The optimal number of clusters (K) if detected, otherwise None.
    """
    inertia = []

    # Calculate inertia for each K
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vectors)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.xticks(k_range[::2])  # Show every second tick mark for clarity
    plt.grid(True)

    elbow_k = find_elbow_point(list(k_range), inertia)
    if elbow_k:
        print(f"Elbow point detected at K = {elbow_k}")
        plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at K={elbow_k}')
        plt.legend()
    else:
        print("No clear elbow point detected.")
    plt.show()

    return elbow_k


def plot_silhouette_scores(skill_vectors, k_range=range(2, 25)):
    """
    Plots the Silhouette Score for K-Means clustering to help determine the optimal number of clusters.

    Args:
        skill_vectors (ndarray): The input data to cluster.
        k_range (range): The range of K values to test.

    Returns:
        dict: A dictionary with K values as keys and their corresponding Silhouette Scores as values.
    """
    silhouette_scores = {}

    # Calculate Silhouette Score for each K
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(skill_vectors)
        labels = kmeans.predict(skill_vectors)
        silhouette_scores[k] = silhouette_score(skill_vectors, labels)

    # Plot the Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'bx-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title('Silhouette Score Method for Optimal K')
    plt.grid(True)
    plt.xticks(list(k_range)[::2])  # Show every second tick mark for clarity
    plt.show()

    return silhouette_scores


def perform_kmeans_clustering(skill_vectors, unique_skills_list, K, random_state=42):
    """
    Performs K-Means clustering on the given skill vectors and calculates metrics.

    Args:
        skill_vectors (ndarray): The input data to cluster.
        unique_skills_list (list): List of unique skills corresponding to the vectors.
        K (int): The number of clusters.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - cluster_labels (ndarray): Cluster labels for each skill vector.
            - inertia_value (float): Inertia value for the chosen K.
            - sc_score (float): Silhouette score for the chosen K.
            - skill_to_cluster_id (pd.Series): Mapping of skills to their cluster IDs.
    """
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=K, random_state=random_state, init='k-means++', n_init='auto')
    cluster_labels = kmeans.fit_predict(skill_vectors)

    # Calculate inertia and silhouette score
    inertia_value = kmeans.inertia_
    sc_score = silhouette_score(skill_vectors, cluster_labels)

    # Mapping skill to its cluster ID
    skill_to_cluster_id = pd.Series(cluster_labels, index=unique_skills_list)

    # Print metrics
    print(f"Inertia for K={K}: {inertia_value}")
    print(f"Silhouette Score for K={K}: {sc_score}")

    return cluster_labels, inertia_value, sc_score, skill_to_cluster_id


def print_clusters_and_skills(skill_to_cluster_id, K):
    """
    Prints the clusters and their associated skills.

    Args:
        skill_to_cluster_id (pd.Series): A mapping of skills to their cluster IDs.
        K (int): The number of clusters.

    Returns:
        None
    """
    for cluster_id in range(K):
        print(f"Cluster {cluster_id}:")
        cluster_skills = [skill for skill, cluster in skill_to_cluster_id.items() if cluster == cluster_id]
        print(", ".join(cluster_skills))
        print("-" * 50)

def map_skills_to_clusters(skills_list, skill_to_cluster_id):
    """
    Maps a list of skills to their corresponding cluster label.
    """
    return [skill_to_cluster_id.get(skill) for skill in skills_list if skill in skill_to_cluster_id]


def plot_cluster_heatmap(df_clustered, figsize=(12, 8), cmap="YlGnBu"):
    """
    Plots a heatmap showing the proportional frequency of cluster IDs in each general job classification.

    Args:
        df_clustered (pd.DataFrame): The DataFrame containing 'general job classification' and 'skill_cluster_ids'.
        figsize (tuple): The size of the heatmap figure (default is (12, 8)).
        cmap (str): The colormap for the heatmap (default is "YlGnBu").

    Returns:
        None
    """
    cluster_frequency = df_clustered.explode('skill_cluster_ids') \
                                     .groupby(['general job classification', 'skill_cluster_ids']) \
                                     .size() \
                                     .unstack(fill_value=0)

    # Normalize the counts row-wise (i.e., by 'general job classification')
    # This converts the counts into proportions (percentages) within each row.
    cluster_proportion = cluster_frequency.div(cluster_frequency.sum(axis=1), axis=0)

    # Plot the heatmap using the proportions
    plt.figure(figsize=figsize)
    sns.heatmap(cluster_proportion, annot=False, cmap=cmap, cbar=True, linewidths=0.5)

    plt.title("Proportional Frequency of Cluster IDs in General Job Classifications", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("General Job Classification", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def reduce_dimensionality_pca(vectors: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Reduces the dimensionality of the given vectors to 2D using PCA.

    Args:
        vectors (ndarray): The input high-dimensional vectors.

    Returns:
        ndarray: The 2D reduced vectors.
    """
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(vectors)

    # Create a DataFrame for easy plotting
    pca_df = pd.DataFrame(reduced_vectors, columns=['PC1', 'PC2'])
    # Add the cluster labels to the DataFrame
    pca_df['Cluster'] = cluster_labels

    return pca_df


def plot_2d_clusters_pca(df_to_plot: pd.DataFrame, k_clusters: int):
    """
    Generates a scatter plot to visualize clusters in a 2D space (PCA).

    Args:
        df_to_plot (pd.DataFrame): DataFrame containing the 2D coordinates 
                                   (first two columns) and cluster labels.
        k_clusters (int): The number of clusters (K) found by the clustering algorithm.
    """

    # Check for required columns
    if len(df_to_plot.columns) < 2 or 'Cluster' not in df_to_plot.columns:
        print(f"Error: DataFrame must have at least two coordinate columns and a 'Cluster' column.")
        return

    # Setup and Scatter Plot
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot
    scatter = plt.scatter(
        df_to_plot.iloc[:, 0], 
        df_to_plot.iloc[:, 1],
        c=df_to_plot['Cluster'], 
        cmap='tab20',
        s=50,          # Marker size
        alpha=0.6      # Transparency
    )

    # Use the inferred reduction method in the title
    title = f'K-Means Clusters (K={k_clusters}) visualized with PCA'
    plt.title(title, fontsize=16)
    
    plt.xlabel(df_to_plot.columns[0], fontsize=12)
    plt.ylabel(df_to_plot.columns[1], fontsize=12)

    # Add a legend
    legend1 = plt.legend(
        *scatter.legend_elements(),
        title="Cluster ID",
        loc="upper right" 
    )
    plt.gca().add_artist(legend1)

    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.show()


def most_common_skills_per_cluster(df_clustered, skill_to_cluster_id, top_n=10):
    """
    Identifies the most common skills in each cluster.

    Args:
        df_clustered (pd.DataFrame): The DataFrame containing 'skills' and 'skill_cluster_ids'.
        skill_to_cluster_id (pd.Series): A mapping of skills to their cluster IDs.
        top_n (int): The number of top skills to return for each cluster.

    Returns:
        dict: A dictionary with cluster IDs as keys and lists of the most common skills as values.
    """
    cluster_skill_counter = {cluster_id: Counter() for cluster_id in skill_to_cluster_id.unique()}

    for _, row in df_clustered.iterrows():
        skills = row['skills']
        if not skills:
            continue
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in skill_to_cluster_id:
                cluster_id = skill_to_cluster_id[skill_lower]
                cluster_skill_counter[cluster_id][skill_lower] += 1

    # Get the most common skills for each cluster
    most_common_skills = {
        cluster_id: [skill for skill, _ in counter.most_common(top_n)]
        for cluster_id, counter in cluster_skill_counter.items()
    }

    return most_common_skills


def create_cluster_features(cluster_ids, K):
    """
    Creates a binary feature vector indicating the presence of each cluster ID.
    Args:
        cluster_ids (list): List of cluster IDs associated with a job posting.
        K (int): Total number of clusters.
    """
    features = np.zeros(K)
    for cid in cluster_ids:
        if cid is not None and 0 <= cid < K:
            features[cid] = 1
    return features


def sentence_embedding(sentence: str, max_length: int, embedding_dim: int, embeddings) -> np.ndarray:
    """
    Converts a sentence to a fixed-size vector using provided word embeddings.

    Args:
        sentence (str): The input sentence.
        max_length (int): Maximum number of tokens to use / sequence length.
        embedding_dim (int): Dimensionality of the embeddings.
        embeddings: Embedding object that exposes `.stoi` and `.vectors` (e.g., GloVe from torchtext).

    Returns:
        np.ndarray: A flattened vector representation of the sentence with shape (max_length * embedding_dim,).
    """
    words = str(sentence).split()
    num_words = min(len(words), max_length)
    embedding_sentence = np.zeros((max_length, embedding_dim), dtype=float)

    for i in range(num_words):
        word = words[i]
        if word in embeddings.stoi:  # Check if the word exists in embedding vocabulary
            embedding_sentence[i] = embeddings.vectors[embeddings.stoi[word]]

    return embedding_sentence.flatten()