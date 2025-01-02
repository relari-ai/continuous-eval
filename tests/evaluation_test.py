from continuous_eval.eval import Dataset, EvaluationRunner, SingleModulePipeline
from continuous_eval.eval.tests import GreaterOrEqualThan
from continuous_eval.metrics.retrieval import (
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


def test_evaluation():
    data = [
        {
            "question": "When was the asteroid named after the first Israeli astronaut for NASA discovered?",
            "retrieved_contexts": [
                "51827 Laurelclark (2001 OH ) is an asteroid named for astronaut Laurel Clark, who was killed in the STS-107 (Columbia) space shuttle reentry disaster on February 1, 2003. 51827 Laurelclark was discovered on July 20, 2001 at Palomar Observatory by the JPL Near Earth Asteroid Tracking Program.",
                "Columbia: The Tragic Loss is a 2004 documentary about the first Israeli astronaut, Ilan Ramon, who died when the Columbia spacecraft disintegrated upon reentry into the Earth's atmosphere. Two months after the disaster, Ramon's diary was found at one of the crash sites and was reconstructed by the Israel Museum along with Israeli police. Interviews with NASA officials and with Ilan's family offer both expert analysis of the flight and a personal look at the tragedy.",
                "51828 Ilanramon (2001 OU ) is an asteroid named for astronaut Ilan Ramon, who was killed in the STS-107 (Columbia) space shuttle reentry disaster on February 1, 2003. 51828 Ilanramon was discovered on July 20, 2001 at Palomar Observatory by the JPL Near Earth Asteroid Tracking Program.",
            ],
            "ground_truth_contexts": [
                "Ilan Ramon (Hebrew: \u05d0\u05d9\u05dc\u05df \u05e8\u05de\u05d5\u05df\u200e , born Ilan Wolferman; June 20, 1954 \u2013 February 1, 2003) was an Israeli fighter pilot and later the first Israeli astronaut for NASA.",
                "51828 Ilanramon (2001 OU ) is an asteroid named for astronaut Ilan Ramon, who was killed in the STS-107 (Columbia) space shuttle reentry disaster on February 1, 2003. 51828 Ilanramon was discovered on July 20, 2001 at Palomar Observatory by the JPL Near Earth Asteroid Tracking Program.",
            ],
            "ground_truths": ["July 20, 2001"],
        },
        {
            "question": "Sean Tully is a fictional character portrayed by Antony Cotton in what British soap opera?",
            "retrieved_contexts": [
                'Sean Tully is a fictional character from the British ITV soap opera, "Coronation Street". Portrayed by Antony Cotton, the character first appeared on 13 July 2003 for one episode, before returning full-time on 12 April 2004.',
                'Clare Devine (also Black and Cunningham) is a fictional character from the British Channel 4 soap opera, "Hollyoaks", played by actresses Gemma Bissix and Samantha Rowley. Bissix agreed to reprise the role in 2009 for the culmination of Warren Fox (Jamie Lomas) and Justin Burton\'s (Chris Fountain) storylines. She later returned to the show in 2013. Clare was killed-off in October 2013 and Bissix said that it would allow the "Hollyoaks" to develop other villainous characters. Bissix has won three British Soap Awards for her portrayal of Clare. She has also been named one of the best British soap opera characters.',
                'Anthony "Tony" Gordon is a fictional character from the British ITV soap opera, "Coronation Street", portrayed by actor Gray O\'Brien. The character first appeared on-screen on 16 September 2007. He appeared as a regular character for two years before departing on 11 December 2009 after being imprisoned for the murder of Liam Connor (Rob James-Collier). He returned on 28 May 2010 planning to escape from prison with the help of his cell mate Robbie Sloan (James Fleet). The character departed once again on 9 June 2010 after being killed-off at the conclusion of his storyline. He was a local businessman, disliked by many of Weatherfield\'s residents for his ruthlessness. His storylines revolved around his business deals, his relationships with Carla Connor and Maria Connor, and the murder of Liam Connor. Tony was named "Bad Boy" of 2009 at the "All About Soap" Bubble awards, and "Villain of the Year" at The British Soap Awards 2009.',
                'Antony Cotton (born Antony Dunn; 5 August 1975) is an English actor, best known for his roles in "Coronation Street" and the original UK version of "Queer as Folk". In March 2013, he won "Let\'s Dance for Comic Relief", defeating fellow finalist Jodie Prenger.',
            ],
            "ground_truth_contexts": [
                'Sean Tully is a fictional character from the British ITV soap opera, "Coronation Street". Portrayed by Antony Cotton, the character first appeared on 13 July 2003 for one episode, before returning full-time on 12 April 2004.',
                'Antony Cotton (born Antony Dunn; 5 August 1975) is an English actor, best known for his roles in "Coronation Street" and the original UK version of "Queer as Folk". In March 2013, he won "Let\'s Dance for Comic Relief", defeating fellow finalist Jodie Prenger.',
            ],
            "ground_truths": ['"Coronation Street"'],
        },
        {
            "question": ' "This Is What You Came For" is a song by Scottish DJ and record producer Calvin Harris, featuring Barbadian singer Rihanna, Rihanna and Harris had previously collaborated on her sixth studio album, serving as the fifth single, Where Have You Been, released in what year? ',
            "retrieved_contexts": [
                '"Where Have You Been" is a song by Barbadian singer Rihanna, from her sixth studio album, "Talk That Talk" (2011) serving as the fifth single. The song was written by Ester Dean, Geoff Mack, Lukasz "Dr. Luke" Gottwald, Henry "Cirkut" Walter, and Calvin Harris, with production handled by the latter three. "Where Have You Been" was released as the third international single from the album on May 8, 2012. The track is a dance-pop and techno house song that draws influence from trance, R&B and hip hop. It is backed by "hard, chilly synths" and contains an electro-inspired breakdown sequence. The song\'s lyrics interpolate Geoff Mack\'s 1959 song "I\'ve Been Everywhere" and speak of a woman who is searching for a partner who will sexually please her.',
                '"Open Wide" is a song by Scottish DJ and producer Calvin Harris from his fourth studio album, "Motion" (2014). It features American rapper Big Sean. Originally released a promotional single on 27 October 2014, the song officially impacted rhythmic contemporary radio in the United States on 27 January 2015 as the album\'s fifth single. "Open Wide" is the vocal version of Harris\'s instrumental track "C.U.B.A", which appears as a B-side to his single "Blame". It peaked at number 23 in the UK, becoming Harris\'s first single to miss the top 10 since 2010.',
                'Mexico Airplay is a record chart published weekly by "Billboard" magazine for singles receiving airplay in Mexico. According to "Billboard"\' s electronic database, the first chart was published on October 1, 2011 with "Give Me Everything" by Cuban-American rapper Pitbull featuring Ne-Yo, Afrojack and Nayer, at number-one. The track also peaked at the top of the American "Billboard" Hot 100. The same year, American performers Maroon 5 featuring Christina Aguilera also peaked at number-one in Mexico and in the United States with "Moves like Jagger". In 2012, Mexican band Jesse & Joy peaked at number one on this chart and the Mexican Espanol Airplay with the song "\u00a1Corre!" that also won the Latin Grammy Awards for Record of the Year and Song of the Year in 2012. Two songs performed by Barbadian singer Rihanna reached number-one, "We Found Love" and "Where Have You Been", the former also was a number-one song in the "Billboard" Hot 100 and its music video won the MTV Video Music Award for Video of the Year, while the latter was nominated for a Grammy Award for Best Pop Solo Performance. "Bailando" by Spanish singer-songwriter Enrique Iglesias reached number-one on the Mexico Airplay, Mexican Espanol Airplay, and the "Billboard" Latin Songs chart in the United States, where it spent 41 consecutive weeks at the top and won the Latin Grammy Award for Song of the Year. In 2015, "Lean On" by American electronic duo Major Lazer and DJ Snake featuring M\u00d8 peaked at number-one on the chart and was named by Spotify as the most streamed song of all time, with 526 million streams globally. By 2016, Scottish DJ Calvin Harris is the act with the most number-one singles on the Mexico Airplay chart, with six chart toppers.',
                '"Thinking About You" is a song by Scottish DJ and record producer Calvin Harris, featuring Jordanian singer Ayah Marar. It was released on 2 August 2013 as the eighth and final single from Harris\' third studio album, "18 Months" (2012). The song was written by Harris and Marar, who previously worked together on Harris\'s promotional single, "Let Me Know" (2004) and "Flashback" (2009), the third single from his second studio album, "Ready for the Weekend".',
            ],
            "ground_truth_contexts": [
                '"Where Have You Been" is a song by Barbadian singer Rihanna, from her sixth studio album, "Talk That Talk" (2011) serving as the fifth single. The song was written by Ester Dean, Geoff Mack, Lukasz "Dr. Luke" Gottwald, Henry "Cirkut" Walter, and Calvin Harris, with production handled by the latter three. "Where Have You Been" was released as the third international single from the album on May 8, 2012. The track is a dance-pop and techno house song that draws influence from trance, R&B and hip hop. It is backed by "hard, chilly synths" and contains an electro-inspired breakdown sequence. The song\'s lyrics interpolate Geoff Mack\'s 1959 song "I\'ve Been Everywhere" and speak of a woman who is searching for a partner who will sexually please her.',
                '"This Is What You Came For" is a song by Scottish DJ and record producer Calvin Harris, featuring Barbadian singer Rihanna. The song was released on 29 April 2016, through Columbia Records and Westbury Road. Featuring influences of house music, Harris produced the song and co-wrote it with Taylor Swift. Rihanna and Harris had previously collaborated on her sixth studio album, "Talk That Talk", which included the international chart-topper "We Found Love" and US top five single "Where Have You Been", the former of which was written and produced by Harris. He played the final version for Rihanna at the 2016 Coachella Music Festival.',
            ],
            "ground_truths": ["2011"],
        },
        {
            "question": "Who developed the American fantasy police comedy-drama which premiered on Fox in January 2016, the songs and artists of which were published by Moraine Music Group, a Nashville independent publisher?",
            "retrieved_contexts": [
                'Samors has authored, co-authored and/or published twenty five books about Chicago\'s neighborhoods, downtown, Michigan Avenue, the Chicago River, Lake Shore Drive, and Chicago\'s airports, and in addition has written and published nostalgic books about growing up in Chicago in the eras of the 1930s, 1940s, 1950s, and 1960s. His book, co-authored with Michael Williams, "The Old Chicago Neighborhood: Remembering Chicago in the 1940s", won the 2003 Independent Publisher Book Award first place award in history, his book "Chicago in the Sixties: Remembering a Time of Change" won the 2007 Independent Publisher Book first place award in history, his book, "Downtown Chicago in Transition", co-authored with Eric Bronsky, won the 2008 Independent Publisher Book second place award for Midwest Region books, and his book, "The Rise of The Magnificent Mile," (co-authored with Eric Bronsky), won the 2009 Independent Publisher Book first place award in the Great Lakes region. He co-authored and published three new books in 2008, including "Clark Weber\'s Rock and Roll Radio", by Clark Weber, "Never Put Ketchup On A Hot Dog", by Bob Schwartz, and "The Rise of the Magnificent Mile", co-authored with Eric Bronsky. In 2010, he published and/or co-authored, with Tony Macaluso and Julia S. Bachrach, "Sounds of Chicago\'s Lakefront: A Celebration of the Grant Park Music Festival, "A Kid From The Windy City," co-authored by Lee B. Stern and Neal Samors, and "Paths Through The Wilderness: American Indian Trail Marker Trees" by Dennis Downes, with Neal Samors. Next, he was the publisher and co-author of "Chicago\'s Lake Shore Drive: Urban America\'s Most Beautiful Roadway" and served as publisher of "Chicago From The Sky: A Region Transformed" by Lawrence Okrent. In 2011, he published and co-authored, "Chicago\'s Classic Restaurants: Past, Present and Future" with Eric Bronsky and Robert Dauber, in 2013 he published and co-authored "Chicago\'s River At Work And At Play" with Steven Dahlman, and, in 2015, he published and co-authored "Now Arriving: Traveling To And From Chicago By Air, 90 Years of Flight" with Christopher Lynch. Dr. Samors publishes books through his company, Chicago\'s Books Press, an imprint of Chicago\'s Neighborhoods, Inc. He has a PhD and MALS from Northwestern University, an MA from Northern Illinois University and a BA from the University of Wisconsin\u2013Madison. In 2010, Dr. Samors was selected as a Prominent Alumnus by the Sullivan High School Alumni Association.',
                'Moraine Music Group is one of Nashville\'s leading independent publishers, with a reputation for unique songs that result in career-making hit singles. For a relatively small company, Moraine\'s songs have appeared on numerous multi-million selling albums spanning various genres of music. Moraine has received over 50 music publishing awards, including 1998 SESAC Publisher of the Year and numerous Canadian Country Music Association Awards for Song, Single, Video and Album of the Year. Moraine\u2019s songs and artists have been included in feature films, television shows, and advertisements including Gareth Dunlop and SHEL\'s "Hold On" in "The Best of Me", SHEL\'s "I Was Born A Dreamer" in a Toys R Us Christmas advertisement, and Gareth Dunlop\'s "Devil Like You" in "Lucifer".',
                'Gareth Dunlop (born East Belfast) is a singer-songwriter from Northern Ireland who is a multi-instrumentalist, engineer and producer. His distinct songwriting and vocal styles has led to his songs being featured in numerous television shows, films and commercials. In 2013, Dunlop\'s song "Wrap Your Arms Around Me" was part of the soundtrack for the movie "Safe Haven" based on a romance novel by Nicholas Sparks. His song was the second best selling single on the album. Currently, Dunlop is in a joint venture publishing deal with Moraine Music Group and Nettwerk One Music. He spends time in both the US and Northern Ireland, but permanently lives in Belfast and operates a recording studio in Hollywood.',
                'Lucifer is an American fantasy police procedural comedy-drama television series developed by Tom Kapinos that premiered on Fox on January 25, 2016. It features a character created by Neil Gaiman, Sam Kieth, and Mike Dringenberg taken from the comic book series "The Sandman", who later became the protagonist of the spin-off comic book series "Lucifer" written by Mike Carey, both published by DC Comics\' Vertigo imprint.',
            ],
            "ground_truth_contexts": [
                'Moraine Music Group is one of Nashville\'s leading independent publishers, with a reputation for unique songs that result in career-making hit singles. For a relatively small company, Moraine\'s songs have appeared on numerous multi-million selling albums spanning various genres of music. Moraine has received over 50 music publishing awards, including 1998 SESAC Publisher of the Year and numerous Canadian Country Music Association Awards for Song, Single, Video and Album of the Year. Moraine\u2019s songs and artists have been included in feature films, television shows, and advertisements including Gareth Dunlop and SHEL\'s "Hold On" in "The Best of Me", SHEL\'s "I Was Born A Dreamer" in a Toys R Us Christmas advertisement, and Gareth Dunlop\'s "Devil Like You" in "Lucifer".',
                'Lucifer is an American fantasy police procedural comedy-drama television series developed by Tom Kapinos that premiered on Fox on January 25, 2016. It features a character created by Neil Gaiman, Sam Kieth, and Mike Dringenberg taken from the comic book series "The Sandman", who later became the protagonist of the spin-off comic book series "Lucifer" written by Mike Carey, both published by DC Comics\' Vertigo imprint.',
            ],
            "ground_truths": ["Tom Kapinos"],
        },
        {
            "question": "What country did the project that Girish Wagh was a key figure of launch in?",
            "retrieved_contexts": [
                "Girish and The Chronicles (commonly abbreviated as GATC) is an Indian Hard rock/Heavy Metal band from Gangtok, Sikkim, formed in 2009, by the Singer-Songwriter/Vocalist Girish Pradhan. Presently based in Bengaluru, Karnataka, GATC is a four-member band, known for their electrifying live shows. GATC has been touring the country and overseas for more than 5 years now, although the line-up has existed since 2006 but was known as Revolving Barrel. After having released numerous singles online since 2009, The band released its first official album on June 2014 under Universal Music Group. The band is known for its peculiar Classic Hard Rock/Heavy Metal sound influenced from the bands of'70s/'80s era. GATC is known to be the first and the only band from Sikkim to ever have travelled/performed overseas and toured on a national scale.",
                "The Second Launch Pad of the Satish Dhawan Space Centre is a rocket launch site in Sriharikota, India. It is the second of two launch pads at the centre. The Second Launch Pad or SLP was designed, supplied, erected & commissioned by MECON Limited, a Government of Indian Enterprise, located at Ranchi (Jharkhand, India) during the period March 1999 to December 2003. It cost about Rs 400 Cr at that time. The second launch pad with associated facilities was built in 2005. However it became operational only on 5 May 2005 with the launching of PSLV-C6. MECON's sub-contractors for this project including Inox India, HEC, Tata Growth, Goderej Boyce, Simplex, Nagarjuna Construction, Steelage, etc. The other Launch Pad being the First Launch Pad. It is used by Polar Satellite Launch Vehicles and Geosynchronous Satellite Launch Vehicles, and is intended for use with future Indian rockets including the Geosynchronous Satellite Launch Vehicle Mk.III",
                "The Tata Nano is a city car manufactured by Tata Motors. Made and sold in India. The Nano was initially launched with a price of one lakh rupees or , which has increased with time. Designed to lure India's burgeoning middle classes away from motorcycles, it received much publicity.",
            ],
            "ground_truth_contexts": [
                "Girish Wagh is currently the Head, Commercial Vehicle Business Unit. He earlier worked in the capacity of Sr. Vice President(Tata Small & Passenger Car Segment) & Head Project Planning and Program Management of Tata Motors. He is a key figure in the Tata Nano's project.",
                "The Tata Nano is a city car manufactured by Tata Motors. Made and sold in India. The Nano was initially launched with a price of one lakh rupees or , which has increased with time. Designed to lure India's burgeoning middle classes away from motorcycles, it received much publicity.",
            ],
            "ground_truths": ["India"],
        },
    ]

    dataset = Dataset.from_data(data)
    module_name = "__test__"

    pipeline = SingleModulePipeline(
        name=module_name,
        dataset=dataset,
        eval=[
            PrecisionRecallF1().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
            RankedRetrievalMetrics().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
        ],
        tests=[
            GreaterOrEqualThan(
                test_name="Recall", metric_name="context_recall", min_value=0.8
            ),
        ],
    )

    # We start the evaluation manager and run the metrics
    runner = EvaluationRunner(pipeline)
    eval_results = runner.evaluate()
    agg = eval_results.aggregate()
    test_results = runner.test(eval_results)

    assert module_name in eval_results.samples
    assert "PrecisionRecallF1" in eval_results.samples[module_name]
    assert "RankedRetrievalMetrics" in eval_results.samples[module_name]
    assert len(eval_results.samples[module_name]["PrecisionRecallF1"]) == len(
        data
    )
    assert len(eval_results.samples[module_name]["PrecisionRecallF1"]) == len(
        data
    )

    metrics = pipeline.module_by_name(module_name).eval
    assert metrics is not None
    schemas = [m.schema for m in metrics]
    keys = [k for sk in schemas for k in sk.keys()]
    assert all(k in agg[module_name] for k in keys)

    assert module_name in test_results.results
    assert "Recall" in test_results.results[module_name]
    assert not test_results.results[module_name]["Recall"]
