import itertools
import logging
import random

import numpy as np
from langchain_community.vectorstores import VectorStore
from tqdm import tqdm

from continuous_eval.llm_factory import LLMFactory, LLMInterface
from continuous_eval.utils.telemetry import telemetry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

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
        vector_store_index: VectorStore,
        generator_llm: LLMInterface = LLMFactory("gpt-3.5-turbo-1106"),
    ):
        if isinstance(vector_store_index, VectorStore):
            if not (
                hasattr(vector_store_index, "similarity_search_by_vector")
                or hasattr(vector_store_index, "similarity_search_by_vector_with_score")
            ):
                raise ValueError(
                    "VectorDB error:"
                    "Could be not find similarity_search_by_vector(_with_score) for vector_store_index. "
                    "Check: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores "
                )
            self.vector_store_index = vector_store_index
        else:
            raise ValueError(
                f"Only Langchain VectorStore is supported for Simple Dataset Generator.\
                             Check: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores"
            )
        assert isinstance(generator_llm, LLMInterface), "generator_llm must be an instance of LLMInterface"
        self._llm = generator_llm

    def _sample_from_vectorstore(self, embedding_vector_size: int, num_seed_vectors: int = 1, top_k: int = 3):
        # Sample from vectorstore based on random vectors
        random_vectors = np.random.rand(num_seed_vectors, embedding_vector_size)
        sampled_chunks = []
        for rv in random_vectors:
            try:
                docs = self.vector_store_index.similarity_search_by_vector(
                    embedding=rv.tolist(),
                    k=top_k,
                )
            except Exception:
                try:
                    docs_scores = self.vector_store_index.similarity_search_by_vector_with_score(
                        embedding=rv.tolist(),
                        k=top_k,
                    )
                    docs = [d for d, _ in docs_scores]
                except Exception as e:
                    raise RuntimeError(f"Failed to sample from vectorstore: {e}")
            # Do not add if sampled aleady
            sampled_chunks.extend([doc for doc in docs if doc not in sampled_chunks])
        return sampled_chunks

    def _generate_q_a(self, chunks, multi_hop: bool, questions_to_generate: int):
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

                multi_hop_prompt_list = [
                    ("Multi Hop Reasoning", MULTI_HOP_REASONING_PROMPT),
                    ("Multi Hop Fact Seeking", MULTI_HOP_FACT_PROMPT),
                ]
                multi_hop_prompt_weights = [0.5, 0.5]

                prompt_type, system_prompt = random.choices(multi_hop_prompt_list, multi_hop_prompt_weights)[0]

                prompt = {
                    "system_prompt": (system_prompt),
                    "user_prompt": (context + "\nQuestion: "),
                }
                prompt_list.append({"prompt": prompt, "question_type": prompt_type})
        else:  # single hop
            for c in chunks:
                context = c.page_content
                single_hop_prompt_list = [
                    ("Single Hop Reasoning", SINGLE_HOP_REASONING_PROMPT),
                    ("Single Hop Fact Seeking", SINGLE_HOP_FACT_PROMPT),
                ]
                single_hop_prompt_weights = [0.5, 0.5]
                prompt_type, system_prompt = random.choices(single_hop_prompt_list, single_hop_prompt_weights)[0]
                prompt = {
                    "system_prompt": (system_prompt),
                    "user_prompt": ("Context:\n" + context + "\nQuestion: "),
                }
                prompt_list.append({"prompt": prompt, "question_type": prompt_type})

        q_a_c_list = []

        for i in range(len(prompt_list)):
            try:
                question = self._llm.run(
                    prompt=prompt_list[i]["prompt"],
                    temperature=0.9,
                )
                if "generation error" in question.lower():
                    raise ValueError(f"Failed to generate question based on prompt {prompt_list[i]['prompt']}")

                context_list = multi_hop_chunk_list[i] if multi_hop else [chunks[i]]
                context_texts = []
                context_metadata = []
                for c in context_list:
                    # Filter out irrelevant sentences
                    extracted_context = self._llm.run(
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

                answer = self._llm.run(
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
            except Exception:
                continue
        return q_a_c_list

    def generate(
        self,
        embedding_vector_size: int,
        num_questions: int = 10,
        multi_hop_percentage: float = 0.2,
        max_try_ratio: int = 3,
        num_seed_vectors: int = 1,
        progress_bar: bool = True,
    ):
        assert embedding_vector_size > 0, "embedding_vector_size must be positive"
        assert num_questions > 0, "num_questions must be positive"
        assert multi_hop_percentage >= 0 and multi_hop_percentage <= 1, "multi_hop_percentage must be in [0, 1]"
        assert max_try_ratio > 0, "max_try_ratio must be positive"
        assert num_seed_vectors > 0, "num_seed_vectors must be positive"

        telemetry.log_event("simple_dataset_generation", f"num_questions: {num_questions}")

        multi_hop_questions = []
        single_hop_questions = []
        num_single_hop_tries = 0
        num_multi_hop_tries = 0

        pbar = tqdm(total=num_questions, desc="Samples", disable=not progress_bar)

        multi_hop_target = int(num_questions * multi_hop_percentage)
        while len(multi_hop_questions) < multi_hop_target:
            pbar.update(len(multi_hop_questions) - pbar.n)
            if num_multi_hop_tries >= multi_hop_target * max_try_ratio:
                logging.info(
                    f"Generated {len(multi_hop_questions)} multi hop questions after {num_multi_hop_tries} tries."
                )
                break
            try:
                chunks = self._sample_from_vectorstore(
                    embedding_vector_size=embedding_vector_size,
                    num_seed_vectors=num_seed_vectors,
                    top_k=3,
                )
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

        single_hop_target = num_questions - len(multi_hop_questions)
        while len(single_hop_questions) < single_hop_target:
            pbar.update(len(multi_hop_questions) + len(single_hop_questions) - pbar.n)
            if num_single_hop_tries >= single_hop_target * max_try_ratio:
                print(f"Generated {len(single_hop_questions)} single hop questions after {num_single_hop_tries} tries.")
                break
            try:
                chunks = self._sample_from_vectorstore(
                    embedding_vector_size=embedding_vector_size,
                    num_seed_vectors=num_seed_vectors,
                    top_k=1,
                )
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

        pbar.close()
        dataset = single_hop_questions + multi_hop_questions
        for idx, d in enumerate(dataset):
            d["uid"] = idx
        if len(dataset) < num_questions:
            raise Warning(f"Could not generate enough questions. Generated {len(dataset)} questions.")

        return dataset
