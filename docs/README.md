# NDArray Documentation

This directory contains the VitePress documentation for the NDArray PHP library.

## Structure

```
docs/
├── .vitepress/          # VitePress configuration
│   └── config.ts       # Site configuration
├── guide/              # User guides
│   ├── getting-started/# Installation, quick start, migration
│   ├── fundamentals/   # Arrays, data types, indexing, views
│   └── operations/     # Creation, arithmetic, math, reductions
├── api/                # API Reference
├── examples/           # Cookbook examples
├── public/             # Static assets (logo, etc.)
└── index.md            # Home page
```

## Development

### Install Dependencies

```bash
npm install
```

### Start Development Server

```bash
npm run docs:dev
```

The documentation will be available at `http://localhost:5173`

### Build for Production

```bash
npm run docs:build
```

Output will be in `docs/.vitepress/dist/`

### Preview Production Build

```bash
npm run docs:preview
```

## Adding Documentation

### Adding a Guide Page

1. Create a new `.md` file in the appropriate directory
2. Add the page to the sidebar in `.vitepress/config.ts`
3. Follow the existing style and formatting

### Adding an API Page

1. Document methods with clear signatures
2. Include parameter descriptions
3. Provide practical examples
4. Link to related methods

### Style Guide

- Use clear, concise language
- Include code examples for all features
- Use admonitions (::: tip, ::: warning) for important notes
- Follow PHP naming conventions in examples
- Show expected output where helpful

## Deployment

The documentation is automatically deployed to GitHub Pages when pushing to the main branch.

## Contributing

When adding new features to the library, please also update the documentation:

1. Update relevant guide pages
2. Add API documentation
3. Include examples showing the new feature
4. Update the changelog
