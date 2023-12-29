import itertools
import random
import re

import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, VectorStore
from tqdm import tqdm

from continuous_eval.dataset import Dataset
from continuous_eval.llm_factory import LLMFactory

MULTI_HOP_REASONING_PROMPT = """You're designing a Q&A system simulation focused on reasoning questions.
Create 1, concise question (under 15 words) that a user might ask, that can be answered relying on information in not just one but all provided context.
Assume that the user is unaware of any specific context, so avoid phrases like "According to context / given provided context...".
Focus on generate a 'Why / How" type of question rather than simple fact-seeking question.
Make sure the context has enough information to provide a full answer to the question.
If the context is not enough to generate a reasonable question, respond with exactly "Generation Error".
"""

MULTI_HOP_FACT_PROMPT = """You're designing a Q&A system simulation focused on fact-seeking questions.
Create 1, concise question (under 15 words) that a user might ask, that can be answered relying on information in not just one but all provided context.
The user is unaware of any specific context, so avoid phrases like "According to the context / passage...".
Make sure the context has enough information to provide a full answer to the question.
If the context is not enough to generate a reasonable question, respond with exactly "Generation Error".
"""

SINGLE_HOP_REASONING_PROMPT = """You're designing a Q&A system simulation focused on reasoning questions.
Create 1, concise question (under 10 words) that a user might ask, that can be answered using the context.
Assume that the user is unaware of any specific context, so avoid phrases like "According to context / given provided context...".
Focus on generate a 'Why / How' type of question rather than simple fact-seeking question.
Make sure the context has enough information to provide a full answer to the question.
If the context is not enough to generate a reasonable question, respond with exactly "Generation Error".
"""

SINGLE_HOP_FACT_PROMPT = """You're designing a Q&A system simulation focused on fact-seeking questions.
Create 1, concise question (under 10 words) that a user might ask, that can be directly answered using the context.
The user is unaware of any specific context, so avoid phrases like "According to the context / passage...".
Make sure the context has enough information to provide a full answer to the question.
If the context is not enough to generate a reasonable question, respond with exactly "Generation Error".
"""

CONTEXT_EXTRACTION_PROMPT = """Extract sentences or paragraphs from following context that are relevant to answering the question.
You are not allowed to change any part of the extracted components.
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
        processor_llm=None,
    ):
        if isinstance(vector_store_index, VectorStore):
            self.vector_store_index = vector_store_index
        else:
            raise ValueError(
                f"Only Langchain VectorStore is supported for Simple Dataset Generator.\
                             Check: https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores"
            )
        self.generator_llm = generator_llm
        self.processor_llm = processor_llm

    def _sample_from_vectorstore(self, index_dimension, num_seed_vectors=1, top_k=4):
        # randomly sample from vectorstore
        # return a set of unique chunks (may be less than num_chunks)
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

    @staticmethod
    def _unpack_qa_string(qa_string):
        # Use regular expressions to find the question and answer
        question_match = re.search(r"Question: (.+?)\n(?:Answer:|$)", qa_string, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"Answer: (.+?)\n|Answer: (.+?)$", qa_string, re.IGNORECASE | re.DOTALL)

        # Extract the question and answer from the matches
        question = question_match.group(1) or question_match.group(2)
        answer = answer_match.group(1) or answer_match.group(2)

        question = question.strip(" \n")
        answer = answer.strip(" \n")

        return {"question": question, "answer": answer}

    def _generate_q_a(self, chunks, multi_hop, questions_to_generate, example_questions=None):
        # Generate questions based on the sampled chunks
        # Use the LLM generator and processor
        # Return a list of Questions

        # Generate prompts for multi-hop and single hop questions
        prompt_list = []
        if multi_hop:
            assert len(chunks) >= 2, "Must input more than 2 chunks for multi-hop questions."
            # generate random unique pairs and triplets of chunks
            chunk_pairs = list(itertools.combinations(chunks, 2))
            # chunk_triplets = list(itertools.combinations(chunks, 3))
            # combined_list = chunk_pairs + chunk_triplets
            multi_hop_chunk_list = random.sample(chunk_pairs, questions_to_generate)

            # generate prompts question, answer for each pair and triplet
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
                # print(f"Generated a {prompt_list[i]['question_type']} question: {question}")

                if multi_hop:
                    context_list = multi_hop_chunk_list[i]
                else:
                    context_list = [chunks[i]]
                # print(f"Processing context: {context_list}")
                context_texts = []
                context_metadata = []
                for c in context_list:
                    # extracted_context = c.page_content

                    # Using LLM to filter out irrelevant sentences
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
                # print(f"Extracted context: {context_texts}")

                answer = llm_generator.run(
                    prompt={
                        "system_prompt": (ANSWER_PROMPT),
                        "user_prompt": (
                            "Context:\n" + "\n".join(context_texts) + "\nQuestion:\n" + question + "\nAnswer: "
                        ),
                    },
                    temperature=0,
                )
                # print(f"Generated answer for question: {answer}")
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
                print(f"Error: {e}")
                continue
        return q_a_c_list

    # @staticmethod
    # def _get_scores(q_a_c_list, processor_llm):
    #     # Get scores for the generated questions
    #     # Use the LLM processor

    #     evaluator_dataset = Dataset(
    #         data =
    #         [
    #             {
    #                 "question": item["question"],
    #                 "retrieved_contexts": item["contexts"],
    #                 "ground_truth_contexts": item["contexts"],
    #                 "answer": item["answer"],
    #                 "ground_truths": [item["answer"]]
    #             }
    #             for item in q_a_c_list
    #         ]
    #     )

    #     retrieval_evaluator = RetrievalEvaluator(
    #         dataset=evaluator_dataset,
    #         metrics=[
    #             PrecisionRecallF1(matching_strategy=MatchingStrategy.EXACT_CHUNK_MATCH),
    #             # LLMBasedContextPrecision(model=processor_llm, log_relevance_by_context=True),
    #         ],
    #     )
    #     generation_evaluator = GenerationEvaluator(
    #         dataset=evaluator_dataset,
    #         metrics=[
    #             DeterministicFaithfulness(),
    #             # LLMBasedFaithfulness(model=processor_llm, classify_by_statement=True),
    #         ],
    #     )
    #     print("Evaluating the generated questions using continuous-eval...")
    #     ret_results=retrieval_evaluator.run()
    #     gen_results=generation_evaluator.run()
    #     combined_results = [{**ret_results[i], **gen_results[i]} for i in range(len(ret_results))]
    #     useful_columns = [
    #         # 'LLM_based_context_precision',
    #         # 'LLM_based_context_relevance_by_context',
    #         # 'LLM_based_faithfulness_score',
    #         'rouge_faithfulness',
    #         "token_overlap_faithfulness"]
    #     combined_results = [{key: combined_results[i][key] for key in useful_columns} for i in range(len(combined_results))]
    #     return combined_results

    # @staticmethod
    # def postprocess(q_a_c_list, processor_llm, examples = None):
    #     # Post-process questions (optional)
    #     if isinstance(q_a_c_list, pd.DataFrame):
    #         q_a_c_list['contexts'] = q_a_c_list['contexts'].apply(lambda x: eval(x))
    #         q_a_c_list['metadata'] = q_a_c_list['metadata'].apply(lambda x: eval(x))
    #         q_a_c_list = q_a_c_list.to_dict(orient="records")

    #     scores = DatasetGenerator._get_scores(q_a_c_list, processor_llm)
    #     q_a_c_list = [{**q_a_c_list[i], **scores[i]} for i in range(len(q_a_c_list))]

    #     return q_a_c_list

    def _generate_prompts_from_examples(self, examples):
        # Generate prompts from examples that contain questions, answers, and contexts (optional)
        # Return a few-shot prompts: {single_hop_few_shot_prompt, multi_hop_few_shot_prompt}
        print("Generating few shot prompts from examples...")
        pass

    def generate(
        self,
        index_dimension,
        num_questions: int = 10,
        examples: list = None,
        multi_hop_percentage: float = 0.2,
        max_try_ratio: int = 3,
        k: int = 3,
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
                # if self.processor_llm:
                #     q_a_c_list = self.postprocess(q_a_c_list, self.processor_llm)
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
                # if self.processor_llm:
                #     q_a_c_list = self.postprocess(q_a_c_list, self.processor_llm)
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
