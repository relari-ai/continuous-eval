import itertools
import random

from langchain.vectorstores import VectorStore

from continuous_eval.llm_factory import LLMFactory

COMMON_RULES = """
The user is unaware of any specific context, so make sure the question makes sense to those who are not aware of the context.
Avoid phrases like "According to context / given provided context..."
Avoid questions that uses references references like he / she / you / they.
Make sure the context has enough information to provide a full answer to the question.
If the context is not enough to generate a reasonable question, respond with exactly "Generation Error".
"""

MULTI_HOP_REASONING_PROMPT = """You're designing a Q&A system simulation focused on reasoning questions.
Create 1, concise question (under 15 words) that a user might ask, that can be answered relying on information in not just one but all provided contexts.
{COMMON_RULES}
"""

MULTI_HOP_FACT_PROMPT = """You're designing a Q&A system simulation focused on fact-seeking questions.
Create 1, concise question (under 15 words) that a user might ask, that can be answered relying on information in not just one but all provided contexts.
{COMMON_RULES}
"""

SINGLE_HOP_REASONING_PROMPT = """You're designing a Q&A system simulation focused on reasoning questions.
Create 1, concise question (under 10 words) that a user might ask, that can be answered using the context.
{COMMON_RULES}
"""

SINGLE_HOP_FACT_PROMPT = """You're designing a Q&A system simulation focused on fact-seeking questions.
Create 1, concise question (under 10 words) that a user might ask, that can be directly answered using the context.
{COMMON_RULES}
"""

CONTEXT_EXTRACTION_PROMPT = """Extract sentences or paragraphs from following context that are relevant to answering the question.
You are not allowed to change any part of the extracted sentences.
If nothing is relevant, respond with exactly "Context Extraction Error".
"""

ANSWER_PROMPT = """Answer the following question using solely the context provided, and not any prior knowledge.
Make sure to use information in each sentence to provide a full answer to the question.
If an answer cannot be reasonably provided using the context alone, respond with exactly "Generation Error".
"""


class SimpleDatasetGenerator:
    def __init__(
        self,
        vector_store_index,
        generator_llm="gpt-3.5-turbo-1106",
        # processor_llm=None,
    ):
        if isinstance(vector_store_index, VectorStore):
            self.vector_store_index = vector_store_index
        else:
            raise ValueError(
                f"Only Langchain VectorStore is supported for Simple Dataset Generator.\
                             Check: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores"
            )
        self.generator_llm = generator_llm

    def _sample_from_vectorstore(self, index_dimension, num_seed_vectors=1, top_k=3):
        # Sample from vectorstore based on random vectors
        random_vectors = []

        for _ in range(num_seed_vectors):
            vector = [random.random() for _ in range(index_dimension)]
            random_vectors.append(vector)

        sampled_chunks = []
        for rv in random_vectors:
            try:
                docs = self.vector_store_index.similarity_search_by_vector(
                    embedding=rv,
                    k=top_k,
                )
            except Exception:
                try:
                    docs_scores = self.vector_store_index.similarity_search_by_vector_with_score(
                        embedding=rv,
                        k=top_k,
                    )
                    docs = [d for d, _ in docs_scores]
                except Exception as e:
                    raise ValueError(
                        f"VectorDB error: {e}.\n\
                        Could be not finding similarity_search_by_vector(_with_score) for vectorDB.\
                        Check: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores"
                    )

            sampled_chunks.extend(docs)

        # remove duplicates
        unique_chunks = []
        for c in sampled_chunks:
            if c not in unique_chunks:
                unique_chunks.append(c)
        return unique_chunks

    def _generate_q_a(self, chunks, multi_hop, questions_to_generate):
        # Generate questions based on the sampled chunks
        prompt_list = []
        if multi_hop:
            assert len(chunks) >= 2, "Must input more than 2 chunks for multi-hop questions."
            # generate random unique pairs from the chunks
            chunk_pairs = list(itertools.combinations(chunks, 2))
            multi_hop_chunk_list = random.sample(chunk_pairs, questions_to_generate)

            for c in multi_hop_chunk_list:

                context = ""
                for i in range(len(c)):
                    context += "Context_" + str(i + 1) + ": " + c[i].page_content + "\n"

                Multi_Hop_Prompt_List = [
                    ("Multi Hop Reasoning", MULTI_HOP_REASONING_PROMPT),
                    ("Multi Hop Fact Seeking", MULTI_HOP_FACT_PROMPT),
                ]

                Multi_Hop_Prompt_Weights = [
                    0.5,
                    0.5,
                ]
                prompt_type, system_prompt = random.choices(Multi_Hop_Prompt_List, Multi_Hop_Prompt_Weights)[0]

                prompt = {
                    "system_prompt": (system_prompt),
                    "user_prompt": (context + "\nQuestion: "),
                }
                prompt_list.append({"prompt": prompt, "question_type": prompt_type})
        else:  # single hop
            for c in chunks:
                context = c.page_content
                Single_Hop_Prompt_List = [
                    ("Single Hop Reasoning", SINGLE_HOP_REASONING_PROMPT),
                    ("Single Hop Fact Seeking", SINGLE_HOP_FACT_PROMPT),
                ]
                Single_Hop_Prompt_Weights = [
                    0.5,
                    0.5,
                ]
                prompt_type, system_prompt = random.choices(Single_Hop_Prompt_List, Single_Hop_Prompt_Weights)[0]
                prompt = {
                    "system_prompt": (system_prompt),
                    "user_prompt": ("Context:\n" + context + "\nQuestion: "),
                }
                prompt_list.append({"prompt": prompt, "question_type": prompt_type})

        q_a_c_list = []
        llm_generator = LLMFactory(model=self.generator_llm)
        for i in range(len(prompt_list)):
            try:
                question = llm_generator.run(
                    prompt=prompt_list[i]["prompt"],
                    temperature=0.9,
                )
                if "generation error" in question.lower():
                    raise ValueError(f"Failed to generate question based on prompt {prompt_list[i]['prompt']}")

                if multi_hop:
                    context_list = multi_hop_chunk_list[i]
                else:
                    context_list = [chunks[i]]
                context_texts = []
                context_metadata = []
                for c in context_list:
                    # Filter out irrelevant sentences
                    extracted_context = llm_generator.run(
                        prompt={
                            "system_prompt": (CONTEXT_EXTRACTION_PROMPT),
                            "user_prompt": (
                                "Context:\n" + c.page_content + "\nQuestion:\n" + question + "\nExtracted Sentences: "
                            ),
                        },
                        temperature=0,
                    )
                    if "context extraction error" in extracted_context.lower():
                        raise ValueError(
                            f"Failed to extract context for question: {question} and context: {c.page_content}"
                        )
                    context_texts.append(extracted_context)
                    context_metadata.append(c.metadata)

                answer = llm_generator.run(
                    prompt={
                        "system_prompt": (ANSWER_PROMPT),
                        "user_prompt": (
                            "Context:\n" + "\n".join(context_texts) + "\nQuestion:\n" + question + "\nAnswer: "
                        ),
                    },
                    temperature=0,
                )
                if "generation error" in answer.lower():
                    raise ValueError(
                        f"Failed to generate answer for question {question}, given context {context_texts}, {context_metadata}"
                    )
                q_a_c_list.append(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": context_texts,
                        "metadata": context_metadata,
                        "question_type": prompt_list[i]["question_type"],
                    }
                )
            except Exception as e:
                # print(f"Error: {e}")
                continue
        return q_a_c_list

    def generate(
        self,
        index_dimension,
        num_questions: int = 10,
        multi_hop_percentage: float = 0.2,
        max_try_ratio: int = 3,
    ):
        multi_hop_target = int(num_questions * multi_hop_percentage)
        single_hop_target = num_questions - multi_hop_target
        multi_hop_questions = []
        single_hop_questions = []
        num_single_hop_tries = 0
        num_multi_hop_tries = 0

        while len(single_hop_questions) < single_hop_target:
            if num_single_hop_tries >= single_hop_target * max_try_ratio:
                print(f"Generated {len(single_hop_questions)} single hop questions after {num_single_hop_tries} tries.")
                break
            try:
                chunks = self._sample_from_vectorstore(index_dimension=index_dimension, num_seed_vectors=1, top_k=1)
                q_a_c_list = self._generate_q_a(
                    chunks=chunks,
                    multi_hop=False,
                    questions_to_generate=1,
                )
                single_hop_questions.extend(q_a_c_list)
                num_single_hop_tries += 1
            except Exception as e:
                print(f"Error: {e}")
                num_single_hop_tries += 1
                continue

        while len(multi_hop_questions) < multi_hop_target:
            if num_multi_hop_tries >= multi_hop_target * max_try_ratio:
                print(f"Generated {len(multi_hop_questions)} multi hop questions after {num_multi_hop_tries} tries.")
                break
            try:
                chunks = self._sample_from_vectorstore(index_dimension=index_dimension, num_seed_vectors=1, top_k=3)
                q_a_c_list = self._generate_q_a(
                    chunks=chunks,
                    multi_hop=True,
                    questions_to_generate=1,
                )
                multi_hop_questions.extend(q_a_c_list)
                num_multi_hop_tries += 1
            except Exception as e:
                print(f"Error: {e}")
                num_multi_hop_tries += 1
                continue

        print(f"{'Results':<10} | {'Target':<10} | {'Generated':<10} | {'Tries':<10}")
        print(f"{'-'*50}")
        print(
            f"{'Single Hop':<10} | {single_hop_target:<10} | {len(single_hop_questions):<10} | {num_single_hop_tries:<10}"
        )
        print(
            f"{'Multi Hop':<10} | {multi_hop_target:<10} | {len(multi_hop_questions):<10} | {num_multi_hop_tries:<10}"
        )
        print(
            f"{'Total':<10} | {num_questions:<10} | {len(single_hop_questions) + len(multi_hop_questions):<10} | {num_single_hop_tries + num_multi_hop_tries:<10}"
        )

        return single_hop_questions + multi_hop_questions
