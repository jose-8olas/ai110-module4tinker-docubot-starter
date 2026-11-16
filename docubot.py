
"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()

        # Build a retrieval index
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")

        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()

                filename = os.path.basename(path)
                docs.append((filename, text))

        return docs

    # -----------------------------------------------------------
    # Text Processing
    # -----------------------------------------------------------

    def tokenize(self, text):
        """
        Convert text into lowercase words and remove punctuation
        and common words that do not add much meaning.
        """
        stop_words = {
            "the", "is", "a", "an", "where", "which",
            "what", "how", "does", "do", "are", "and",
            "or", "to", "in", "of", "for", "with",
            "on", "by", "this", "that", "it", "all"
        }

        words = text.lower().split()
        cleaned_words = []

        for word in words:
            word = word.strip(".,!?():;\"'")

            if word and word not in stop_words:
                cleaned_words.append(word)

        return cleaned_words

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build an inverted index mapping lowercase words to the
        documents they appear in.
        """
        index = {}

        for filename, text in documents:
            words = self.tokenize(text)

            for word in words:
                if word not in index:
                    index[word] = []

                if filename not in index[word]:
                    index[word].append(filename)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score based on the number
        of unique query words found in the text.
        """
        query_words = set(self.tokenize(query))
        text_words = set(self.tokenize(text))

        score = len(query_words.intersection(text_words))

        return score

    def retrieve(self, query, top_k=3):
        """
        Split documents into paragraphs and retrieve the most
        relevant paragraphs instead of entire documents.
        """
        query_words = self.tokenize(query)

        # Find candidate documents using the inverted index
        candidate_filenames = set()

        for word in query_words:
            candidate_filenames.update(self.index.get(word, []))

        results = []

        # Score individual paragraphs in candidate documents
        for filename, text in self.documents:
            if filename in candidate_filenames:

                paragraphs = text.split("\n\n")

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()

                    if not paragraph:
                        continue

                    score = self.score_document(query, paragraph)

                    if score > 0:
                        results.append((score, filename, paragraph))

        # Sort paragraphs by relevance score
        results.sort(key=lambda item: item[0], reverse=True)

        # Return only the top matching paragraphs
        return [
            (filename, paragraph)
            for score, filename, paragraph in results[:top_k]
        ]

    # -----------------------------------------------------------
    # Retrieval Guardrail
    # -----------------------------------------------------------

    def has_meaningful_context(self, query, snippets):
        """
        Check whether the best retrieved paragraph has enough
        matching query terms to be considered relevant.
        """
        if not snippets:
            return False

        best_score = self.score_document(query, snippets[0][1])
        query_word_count = len(set(self.tokenize(query)))

        # Require at least two matching terms for longer queries.
        # For a one-word query, require that one word to match.
        minimum_score = min(2, query_word_count)

        return best_score >= minimum_score

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns relevant paragraphs without using an LLM.
        Refuses when there is insufficient retrieval evidence.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not self.has_meaningful_context(query, snippets):
            return "I do not know based on these docs."

        formatted = []

        for filename, paragraph in snippets:
            formatted.append(f"[{filename}]\n{paragraph}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses retrieval to select paragraphs, then asks Gemini
        to generate an answer using only those paragraphs.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not self.has_meaningful_context(query, snippets):
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive generation only baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
