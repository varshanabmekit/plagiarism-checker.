from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input documents
doc1 = "This project is about sparse matrix storage"
doc2 = "This project explains efficient matrix storage"

# Convert text to matrix
vectorizer = CountVectorizer() 
matrix = vectorizer.fit_transform([doc1, doc2])

# Convert to sparse matrix
sparse_matrix = matrix.toarray()

# Calculate similarity
similarity = cosine_similarity(matrix)[0][1]

# Output result
print("Sparse Matrix:\n", sparse_matrix)
print("Plagiarism Similarity:", similarity * 100, "%")
