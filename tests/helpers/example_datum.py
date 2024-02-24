CAPITAL_OF_FRANCE = {
    "question": "What is the capital of France?",
    "retrieved_context": [
        "Paris is the largest city in France.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_context": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}

ROMEO_AND_JULIET = {
    "question": "Who wrote Romeo and Juliet?",
    "retrieved_context": [
        "Shakespeare was a playwright.",
        "Romeo and Juliet is a play by Shakespeare.",
    ],
    "ground_truth_context": [
        "Shakespeare was a playwright.",
        "Romeo and Juliet is a play by William Shakespeare.",
    ],
    "answer": "William Shakespeare",
    "ground_truths": ["William Shakespeare"],
}


IMPLICATIONS_GLOBAL_WARMING = {
    "question": "What are the implications of global warming?",
    "retrieved_context": [
        (
            "Global warming refers to the long-term rise in the average temperature of the Earth's climate system. "
            "It is a major aspect of climate change, and has been demonstrated by direct temperature measurements "
            "and by measurements of various effects of the warming. The terms are commonly used interchangeably, "
            "though global warming is more specifically about rising surface temperatures, while climate change includes "
            "global warming as well as everything else that increasing greenhouse gas amounts will affect. "
            "A 2016 report stated that the Arctic is warming at a rate double that of the global average. "
            "The effects of global warming include rising sea levels, regional changes in precipitation, more frequent "
            "extreme weather events such as heat waves, and expansion of deserts. Surface temperature increases are "
            "greatest in the Arctic, which has contributed to the retreat of glaciers, permafrost, and sea ice. "
            "Overall, higher temperatures bring more rain and snowfall, but for some regions, droughts and wildfires "
            "increase instead. Climate change threatens to diminish the supply of fresh water. A warming atmosphere "
            "can hold, and more frequently does hold, larger quantities of water vapor, which can lead to more intense "
            "rainstorms, causing destructive erosion. Warming also creates conditions that can lead to more powerful "
            "hurricanes. Rising temperatures also have the potential to change the nature of global rainfall, snow, "
            "and river flows. Effects significant to humans include the threat to food security from decreasing crop "
            "yields and the abandonment of populated areas due to rising sea levels. Because the climate system has "
            "a large inertia and greenhouse gases will remain in the atmosphere for a long time, climatic changes and "
            "their effects will continue for many centuries even if greenhouse gas emissions are stopped."
        ),
        (
            "Environmental impacts of climate change might include harsher hurricanes and storms, the death of reefs "
            "and forests, more frequent and severe droughts, increased heat waves, and stronger, more intense wildfires. "
            "Such changes will have significant implications for human societies and the natural world. The extent of these "
            "effects will depend largely on the degree of future global warming and the strategies adopted for mitigation "
            "and adaptation. Some effects of climate change, such as record high temperatures and melting glaciers, are "
            "already being observed. The world community has taken some steps towards addressing climate change. The "
            "2015 Paris Agreement, for instance, set the goal of limiting global warming to well below 2.0 degrees Celsius "
            "relative to pre-industrial levels; and to limit the increase to 1.5 degrees Celsius, recognizing that this would "
            "substantially reduce the risks and impacts of climate change. This agreement is meant to signal the beginning "
            "of the end of over two centuries of predominance of fossil fuels. Some experts have called for a coordinated "
            "economic transition to rapid decarbonization, climate finance and 'climate justice'. The overall conclusion of "
            "the Intergovernmental Panel on Climate Change (IPCC), the peak scientific body on climate change, is that it "
            "is 'extremely likely' that the majority of global warming since 1950 has been caused by human activities."
        ),
    ],
    "ground_truth_context": [
        (
            "Climate change threatens to diminish the supply of fresh water. A warming atmosphere "
            "can hold, and more frequently does hold, larger quantities of water vapor, which can lead to more intense "
            "rainstorms, causing destructive erosion. To mitigate these impacts, "
            "strategies such as reducing greenhouse gas emissions and enhancing sustainability practices are vital. "
            "The Paris Agreement of 2015 marks a global effort to limit warming and reduce the risks associated with "
            "climate change, aiming to transition away from fossil fuels towards cleaner, renewable sources of energy."
        )
    ],
    "answer": "Reducing greenhouse gas emissions, transitioning to renewable energy",
    "ground_truths": [
        "Reducing greenhouse gas emissions",
        "Transitioning to renewable energy",
    ],
}

FARGO = {
    "question": "Did Fargo win the golden globe nominations for both seasons?",
    "retrieved_context": [
        "Fargo is an American black comedy crime drama television series created and primarily written by Noah Hawley. The show is inspired by the 1996 film of the same name, which was written and directed by the Coen brothers, and takes place within the same fictional universe. The Coens were impressed by Hawley's script and agreed to be named as executive producers.[3] The series premiered on April 15, 2014, on FX,[3] and follows an anthology format, with each season set in a different era and location, with a different story and mostly new characters and cast, although there is minor overlap. Each season is heavily influenced by various Coen brothers films, with each containing numerous references to them.[4]",
        "The first season, set primarily in Minnesota and North Dakota from January 2006 to February 2007 and starring Billy Bob Thornton, Allison Tolman, Colin Hanks, and Martin Freeman, received wide acclaim from critics.[5] It won the Primetime Emmy Awards for Outstanding Miniseries, Outstanding Directing, and Outstanding Casting, and received 15 additional nominations including Outstanding Writing, another Outstanding Directing nomination, and acting nominations for all four leads. It also won the Golden Globe Awards for Best Miniseries or Television Film and Best Actor – Miniseries or Television Film for Thornton.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "ground_truth_context": [
        "The first season, set primarily in Minnesota and North Dakota from January 2006 to February 2007 and starring Billy Bob Thornton, Allison Tolman, Colin Hanks, and Martin Freeman, received wide acclaim from critics.[5] It won the Primetime Emmy Awards for Outstanding Miniseries, Outstanding Directing, and Outstanding Casting, and received 15 additional nominations including Outstanding Writing, another Outstanding Directing nomination, and acting nominations for all four leads. It also won the Golden Globe Awards for Best Miniseries or Television Film and Best Actor – Miniseries or Television Film for Thornton.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "answer": "Berlin",
    "ground_truths": [
        "Yes, they did get a nomination in season 1 and 2.",
        "Not really, they didn't win for season three.",
    ],
}

# =====================================================================================
# CODE METRICS EXAMPLES
# =====================================================================================

PYTHON_CODE_EXAMPLES = [
    {
        "answer": "def function(x, y):\n  return x + y",
        "ground_truths": [
            "def foo(x, y):\n  return x * y",
            "def foo(x, y):\n  return x + y",
        ],
    },
    {
        "answer": "def foo(x, y):\n  print(x + y)",
        "ground_truths": ["def function(x, y):\n  return x + y"],
    },
    {
        "answer": "class MyClass:\n  def __init__(self, x):\n        self.x = x",
        "ground_truths": [
            "class MyClass:\n  def __init__(self, x):\n    self._x = x\n    @property\n    def x(self):\n      return self._x",
        ],
    },
    {
        "answer": "print('Hello, World!')",
        "ground_truths": ["def function(x, y):\n  return x + y"],
    },
    {
        "answer": "function(x, y):\nreturn x + y",
        "ground_truths": ["def function(x, y):\n  return x + y"],
    },
    {
        "answer": "def rotate(text, key):\n    alpha = string.ascii_lowercase\n    alpha_shift = alpha[key:] + alpha[:key]\n    table = str.maketrans(alpha + alpha.upper(), alpha_shift + alpha_shift.upper())\n    return text.translate(table)",
        "ground_truths": [
            "def rotate(text, key):\n    newchars = string.ascii_lowercase[key:] + string.ascii_lowercase[:key]\n    trans = str.maketrans(string.ascii_lowercase + string.ascii_lowercase.upper(), newchars + newchars.upper())\n    return text.translate(trans)"
        ],
    },
]
