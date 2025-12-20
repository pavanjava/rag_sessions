from qdrant_client import QdrantClient, models
import uuid


class ChonkieVectorStore:
    def __init__(self, url="http://localhost:6333", api_key: str = "th3s3cr3tk3y", collection_prefix: str = ""):
        """Initialize the vector store with a Qdrant client connection.

        Args:
            url: URL of the Qdrant server
            collection_prefix: Optional prefix for collection names
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_prefix = collection_prefix

    def create_collection(self, collection_name, vector_size=100,
                          distance=models.Distance.COSINE) -> str:
        """Create a new collection in Qdrant.

        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance: Distance metric to use
        """
        full_collection_name = f"{self.collection_prefix}{collection_name}"
        if not self.client.collection_exists(collection_name=full_collection_name):
            self.client.create_collection(
                collection_name=full_collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )

        return full_collection_name

    def upsert_point(self, collection_name, vector, payload, point_id=None):
        """Insert or update a point in the collection.

        Args:
            collection_name: Name of the collection
            vector: Vector data as list of floats
            payload: Dictionary containing point metadata
            point_id: Optional ID for the point (generates UUID if not provided)

        Returns:
            ID of the inserted point
        """
        if point_id is None:
            point_id = str(uuid.uuid4())


        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector=vector,
                ),
            ],
        )

        return point_id
