import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjax],
  },
	integrations: [
		starlight({
			title: 'Continuous Eval',
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 4, },
			customCss: [
				// Relative path to your custom CSS file
				'./src/styles/custom.css',
			],
			social: {
				github: 'https://github.com/relari-ai/continuous-eval/tree/main',
			},
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Introduction', link: '/getting-started/introduction/'},
						{ label: 'Installation', link: '/getting-started/installation/' },
						{ label: 'Quick Start', link: '/getting-started/quickstart/' },
					],
				},
				{
					label: 'Metrics',
					items: [
						{ label: 'Overview', link: '/metrics/overview/' },
						{
							label: 'Retrieval',
							autogenerate: { directory: '/metrics/Retrieval/' }
						},
						{
							label: 'Generation',
							items: [
								{
									label: 'Deterministic',
									autogenerate: { directory: '/metrics/Generation/Deterministic/' }
								},
								{
									label: 'Semantic',
									items: [
										{ label: 'DeBERTa Answer Scores', link: '/metrics/generation/semantic/deberta_answer_scores/' },
										{ label: 'BERT Answer Similarity', link: '/metrics/generation/semantic/bert_answer_similarity/' },
										{ label: 'BERT Answer Relevance', link: '/metrics/generation/semantic/bert_answer_relevance/' },
									]
								},
								{
									label: 'LLM-Based',
									autogenerate: { directory: '/metrics/Generation/LLM-Based/' }
								},
							]
						},
						{
							label: 'Metric Ensembling',
							items: [
								{ label: 'Classification', link: '/metrics/ensembling/classifier/' },
							],
						},
					],
				},
				{
					label: 'Datasets',
					autogenerate: { directory: 'evaluators' },
				},
        		{
					label: 'Examples',
					autogenerate: { directory: 'examples' },
				},
			],
		}),
	],
});
