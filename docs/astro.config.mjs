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
						{ label: 'Installation', link: '/getting-started/installation/' },
					],
				},
				{
					label: 'Metrics',
					autogenerate: { directory: 'metrics' },
				},
        {
					label: 'Evaluators',
					autogenerate: { directory: 'evaluators' },
				},
        {
					label: 'Classification',
					items: [
						{ label: 'Conformal Prediction', link: '/classification/conformal_prediction/' },
						{ label: 'Classification', link: '/classification/classifier/' },
					],
				},
        {
					label: 'Examples',
					autogenerate: { directory: 'examples' },
				},
			],
		}),
	],
});
