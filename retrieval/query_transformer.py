from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryTransformer:
    def __init__(self,model_name='gpt-4.1-mini'):
        self.llm=ChatOpenAI(
            model=model_name,
            temperature=0
        )

        self.parser=StrOutputParser()

    def rewrite_query(self,query,conversation_history=''):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Rewrite the user's question into one clear,
                standalone search query for a document retrieval system.

                Use conversation history to resolve references
                such as "it", "that", or "the previous topic".

                Preserve the user's original intent.
                Do not answer the question.
                Return only the rewritten query.
                """
            ),
            (
                "human",
                """
                Conversation history:
                {history}

                User question:
                {question}
                """
            )
        ])

        chain= prompt | self.llm | self.parser

        return chain.invoke({
            "history":conversation_history,
            "question": query
        }).strip()

    def generate_multi_queries(self, query, num_queries=3):

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Generate alternative search queries for a
                document retrieval system.

                Each query should explore a different wording
                or relevant aspect of the original question.

                Rules:
                - Preserve the original intent.
                - Do not answer the question.
                - Return exactly the requested number of queries.
                - Return one query per line.
                - Do not include numbering or explanations.
                """
            ),
            (
                "human",
                """
                Original question:
                {question}

                Number of alternative queries:
                {num_queries}
                """
            )
        ])

        chain = prompt | self.llm | self.parser

        response = chain.invoke({
            "question": query,
            "num_queries": num_queries
        })

        queries = [
            line.strip().lstrip("-•").strip()
            for line in response.splitlines()
            if line.strip()
        ]

        # Remove accidental duplicates
        unique_queries = list(dict.fromkeys(queries))

        return unique_queries[:num_queries]
