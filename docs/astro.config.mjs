import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Continuous Eval',
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
				// {
				// 	label: 'Metrics',
				// 	items: [
				// 	  	{ label: 'Overview', link: '/metrics_alt/overview/' },
				// 	  	{
				// 			label: 'Retrieval Metrics',
				// 			link: '/metrics/retrieval_metrics/',
				// 			subItems: [
				// 				{
				// 					label: 'Deterministic',
				// 					link: '/metrics/retrieval_metrics/deterministic/',
				// 					subItems: [
				// 						{ label: 'Sub-item 1', link: '/metrics/retrieval_metrics/deterministic/precision_recall/' },
				// 						{ label: 'Sub-item 2', link: '/metrics/retrieval_metrics/deterministic/ranked_metrics/' },
				// 					],
				// 				},
				// 				{ label: 'LLM-based', link: '/metrics/retrieval_metrics/LLM_Based/' },
				// 			],
				// 		},
				// 		{
				// 			label: 'Generation Metrics',
				// 			link: '/metrics/generation_metrics/',
				// 			subItems: [
				// 				{ label: 'Sub-item 3', link: '/metrics/generation_metrics/sub-item-3/' },
				// 				{ label: 'Sub-item 4', link: '/metrics/generation_metrics/sub-item-4/' },
				// 			],
				// 		},
				// 	],
				//   },
				{ label: 'Metrics',
					autogenerate: { directory: 'metrics' },
				},
				// {
				// 	label: 'Metrics_Alt',
				// 	items: [
				// 		{ label: 'Overview', link: '/metrics/overview/' },
				// 		{ label: 'Retrieval Metrics', link: '/metrics/retrieval_metrics/' },
				// 		{ label: 'Generation Metrics', link: '/metrics/generation_metrics/' },
				// 	],
				// },
				{
					label: 'Metric Ensembling',
					items: [
						{ label: 'Conformal Prediction', link: '/classification/conformal_prediction/' },
						{ label: 'Classification', link: '/classification/classifier/' },
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
