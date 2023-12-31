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
				{
					label: 'Metrics',
					items: [
						{ label: 'Overview', link: '/metrics/overview/' },
						{ label: 'Retrieval Metrics', link: '/metrics/retrieval_metrics/' },
						{ label: 'Generation Metrics', link: '/metrics/generation_metrics/' },
					],
				},
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
