import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';
import astroD2 from 'astro-d2'

// https://astro.build/config
export default defineConfig({
  site: 'https://docs.relari.ai',
  base: '/v0.3',
  outDir: './dist/v0.3',
  trailingSlash: "never",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjax],
  },
	integrations: [
		starlight({
			title: 'Continuous Eval',
      components: {
        // Override the default `SocialIcons` component.
        ThemeSelect: './src/components/ThemeSelect.astro',
      },
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
						{ label: 'Start Here!', link: '/'},
						{ label: 'Why continuous-eval?', link: '/getting-started/introduction/'},
						{ label: 'Installation', link: '/getting-started/installation/' },
						{ label: 'Quick Start', link: '/getting-started/quickstart/' },
					],
				},
        {
					label: 'Orchestration',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Pipeline', link: '/pipeline/pipeline'},
						{ label: 'Dataset', link: '/pipeline/eval_dataset' },
						{ label: 'Metrics and Tests', link: '/pipeline/metrics_and_tests'},
            { label: 'Pipeline Logger', link: '/pipeline/pipeline_logger' },
            { label: 'Evaluation Runner', link: '/pipeline/eval_runner' },
						{ label: 'Evaluation Manager (Deprecated)', link: '/pipeline/eval_manager' },
					],

        },
				{
					label: 'Metrics',
					items: [
						{ label: 'Overview', link: '/metrics/overview/' },
						{
							label: 'Retrieval',
							collapsed: true,
							items: [
								{
									label: 'Deterministic',
									autogenerate: { directory: '/metrics/Retrieval/Deterministic/' }
								},
								{
									label: 'LLM-Based',
									autogenerate: { directory: '/metrics/Retrieval/LLM-Based/' }
								},
							]
						},
						{
							label: 'Text Generation',
							collapsed: true,
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
							label: 'Code Generation',
							collapsed: true,
							items : [
								{
									label: 'Deterministic',
									autogenerate: { directory: '/metrics/Code/Deterministic/' }
								},
								{
									label: 'LLM-Based',
									autogenerate: { directory: '/metrics/Code/LLM-Based/' }
								}
							]
						},
						{
							label: 'Agent Tool Use',
							collapsed: true,
							items : [
								{
									label: 'Deterministic',
									autogenerate: { directory: '/metrics/Tools/Deterministic/' }
								},
							]
						},
						{
							label: 'Classification',
							collapsed: true,
							items : [
								{
									label: 'Deterministic',
									autogenerate: { directory: '/metrics/Classification/Deterministic/' }
								},
							]
						},
						{label: 'Custom Metrics', link: '/metrics/custom_metrics/'},
						{label: 'Metric Ensembling', link: '/metrics/ensembling_classifier/'},
						],
				},
				{
					label: 'Datasets',
					autogenerate: { directory: '/dataset/' }
				},
        		{
					label: 'Examples',
					items : [
						{
							label: 'Basic',
							autogenerate: { directory: '/examples/Basics/' }
						},
						{
							label: 'Advanced',
							autogenerate: { directory: '/examples/Advanced/' }
						},
						{
							label: 'End-to-End',
							autogenerate: { directory: '/examples/End_to_End/' }
						},
					],
				},
			],
		}),
    astroD2({output: 'd2', basePath: '/v0.3'}),
	],
});
