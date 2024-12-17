from continuous_eval.metrics.base.metric import Arg, Field
from continuous_eval.metrics.custom import CustomMetric, Example

# In this example we want to create a custom metric to evaluate the conciseness of a given answer to a question.
# We will use a scale from 1 to 3, let's define an example for each score (we will use to define the metric later)

criteria = "Conciseness in communication refers to the expression of ideas in a clear and straightforward manner, using the fewest possible words without sacrificing clarity or completeness of information. It involves eliminating redundancy, verbosity, and unnecessary details, focusing instead on delivering the essential message efficiently. "
rubric = """Use the following rubric to assign a score to the answer based on its conciseness:
- Score 1: The answer is overly verbose, containing a significant amount of unnecessary information, repetition, or redundant expressions that do not contribute to the understanding of the topic.
- Score 2: The answer includes some unnecessary details or slightly repetitive information, but the excess does not severely hinder understanding.
- Score 3:The answer is clear, direct, and to the point, with no unnecessary words, details, or repetition."""
examples = [
    Example(
        input={
            "question": "What causes sea breezes?",
            "answer": "Sea breezes are caused by the differential heating of land and sea. During the day, land heats up faster than water, creating a pressure difference that drives cooler air from the sea towards the land.",
        },
        output={
            "reasoning": "This answer receives a high score for conciseness. It directly addresses the question without unnecessary details, providing the essential explanation in a clear and straightforward manner.",
            "score": 3,
        },
    ),
    Example(
        {
            "question": "What causes sea breezes?",
            "answer": "Sea breezes are a result of the interesting interplay between the heating rates of land and water. Essentially, during the sunlit hours, land heats up much more rapidly compared to the ocean. This difference in heating leads to a variation in air pressure; as the warmer air over the land rises due to its lower density, a pressure difference is created. Cooler air from the sea, being denser, moves towards the land to balance this pressure difference. However, it’s not just about temperature and pressure; the Earth’s rotation also plays a part in directing the breeze, adding a slight twist to the direction the breeze comes from. This natural phenomenon is quite essential, contributing to local weather patterns and offering relief on hot days along coastal areas.",
        },
        output={
            "reasoning": "This answer would receive a score of 2 for conciseness. It provides a more detailed explanation than necessary for a straightforward question but does not delve into excessive verbosity. The answer introduces the basic concept accurately and includes relevant details about the cause of sea breezes. However, it also incorporates additional information about the Earth's rotation, which, while related, is not strictly necessary to understand the fundamental cause of sea breezes.",
            "score": 2,
        },
    ),
    Example(
        {
            "question": " What causes sea breezes?",
            "answer": "To understand what causes sea breezes, it's important to start by recognizing that the Earth is made up of various surfaces, such as land and water, which both play a significant role in the way our climate and weather patterns are formed. Now, during the daylight hours, what happens is quite fascinating. The sun, which is our primary source of light and heat, shines down upon the Earth's surface. However, not all surfaces on Earth respond to this heat in the same way. Specifically, land tends to heat up much more quickly and to a higher degree compared to water. This discrepancy in heating rates is crucial because it leads to differences in air pressure. Warmer air is less dense and tends to rise, whereas cooler air is more dense and tends to sink. So, as the land heats up, the air above it becomes warmer and rises, creating a kind of vacuum that needs to be filled. Consequently, the cooler, denser air over the water begins to move towards the land to fill this space. This movement of air from the sea to the land is what we experience as a sea breeze. It's a fascinating process that not only demonstrates the dynamic nature of our planet's climate system but also highlights the intricate interplay between the sun, the Earth's surface, and the atmosphere above it.",
        },
        output={
            "reasoning": "This answer would score lower on conciseness. While it is informative and covers the necessary scientific principles, it contains a significant amount of introductory and explanatory material that, while interesting, is not essential to answering the specific question about the cause of sea breezes.",
            "score": 1,
        },
    ),
]

metric = CustomMetric(
    name="Conciseness",
    criteria=criteria,
    rubric=rubric,
    arguments={
        "question": Arg(
            type=str, description="The question to asked to the system."
        ),
        "answer": Arg(type=str, description="The answer to evaluate."),
    },
    response_format={
        "reasoning": Field(
            type=str,
            description="The reasoning for the score given to the answer",
        ),
        "score": Field(
            type=int, description="The score of the answer between 1 and 3"
        ),
    },
)

# Let's calculate the metric for the first datum
datum = {
    "question": "What causes seasons to change?",
    "answer": "The change in seasons is primarily caused by the Earth's tilt on its axis combined with its orbit around the Sun. This tilt leads to variations in the angle and intensity of sunlight reaching different parts of Earth at different times of the year.",
}

print(metric(**datum))
