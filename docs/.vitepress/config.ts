import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'NDArray PHP',
  description: 'High-performance N-dimensional arrays for PHP, powered by Rust',

  base: '/ndarray/',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3c873a' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:site_name', content: 'NDArray PHP' }],
  ],

  themeConfig: {
    logo: '/logo.png',

    // outline: [2, 3],

    nav: [
      { text: 'User Guide', link: '/guide/getting-started/what-is-ndarray' },
      { text: 'API Reference', link: '/api/' },
      {
        text: '0.1.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/contributing' },
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'What is NDArray?', link: '/guide/getting-started/what-is-ndarray' },
            { text: 'Installation', link: '/guide/getting-started/installation' },
            { text: 'Quick Start', link: '/guide/getting-started/quick-start' },
            { text: 'NumPy Migration', link: '/guide/getting-started/numpy-migration' },
          ]
        },
        {
          text: 'Fundamentals',
          collapsed: false,
          items: [
            { text: 'Understanding Arrays', link: '/guide/fundamentals/understanding-arrays' },
            { text: 'Data Types', link: '/guide/fundamentals/data-types' },
            { text: 'Indexing and Slicing', link: '/guide/fundamentals/indexing-and-slicing' },
            { text: 'Iterating Over Arrays', link: '/guide/fundamentals/iteration' },
            { text: 'Broadcasting', link: '/guide/fundamentals/broadcasting' },
            { text: 'Views vs Copies', link: '/guide/fundamentals/views-vs-copies' },
            { text: 'Operations', link: '/guide/fundamentals/operations' },
            { text: 'Printing', link: '/guide/fundamentals/printing' },
          ]
        },
        {
          text: 'Advanced',
          collapsed: false,
          items: [
            { text: 'Performance', link: '/guide/advanced/performance' },
            { text: 'FFI Internals', link: '/guide/advanced/ffi-internals' },
            { text: 'Troubleshooting', link: '/guide/advanced/troubleshooting' },
          ]
        },
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' },
          ]
        },
        {
          text: 'Core',
          items: [
            { text: 'NDArray Class', link: '/api/ndarray-class' },
            { text: 'Array Creation', link: '/api/array-creation' },
            { text: 'Array Manipulation', link: '/api/array-manipulation' },
            { text: 'Indexing Routines', link: '/api/indexing-routines' },
            { text: 'Array Import/Export', link: '/api/array-import-export' },
          ]
        },
        {
          text: 'Operations',
          items: [
            { text: 'Mathematical Functions', link: '/api/mathematical-functions' },
            { text: 'Logic Functions', link: '/api/logic-functions' },
            { text: 'Bitwise Operations', link: '/api/bitwise-operations' },
            { text: 'Statistics', link: '/api/statistics' },
            { text: 'Sorting & Searching', link: '/api/sorting-searching' },
            { text: 'Linear Algebra', link: '/api/linear-algebra' },
          ]
        },
        {
          text: 'Other',
          items: [
            { text: 'Exceptions', link: '/api/exceptions' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/phpmlkit/ndarray' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 CodeWithKyrian'
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/phpmlkit/ndarray/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    lastUpdated: {
      text: 'Updated at',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    }
  }
})
