import { defineCollection } from 'astro:content';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
	docs: defineCollection({ schema: docsSchema() }),
	i18n: defineCollection({ type: 'data', schema: i18nSchema() }),
};

export const base = '/';

export const versions = [
  ['v0.3', 'v0.3.x (canary)'],
  ['v0.2', 'v0.2.x (latest)'],
];

export const defaultVersion = 'v0.2';