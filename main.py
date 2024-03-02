from index_documents import get_indexed_vector_store
from query import run

if __name__ == "__main__":

    answer = run(get_indexed_vector_store(), "What is a generic?")
    print(answer)
