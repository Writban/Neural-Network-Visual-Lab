# Neural Network Visual Lab

An interactive browser application for constructing and comparing small neural
networks on reproducible two-dimensional datasets. The lab trains real models
in the browser and redraws their decision boundaries as their weights change.

## What it does

- compares two independently configured neural networks side by side;
- switches between XOR, nested-circle, and interleaving-moon datasets;
- reproduces datasets and initial weights from numeric seeds;
- edits hidden-layer architecture neuron by neuron, alongside activation
  function and learning rate controls;
- displays live binary cross-entropy loss, training accuracy, clean test
  accuracy, and decision boundaries;
- lets users click a decision map to inspect an individual prediction;
- includes a 32-part, five-chapter guided course that progressively reveals the
  lab, checks understanding, and responds to real learner interactions;
- offers dismissible, segment-specific bonus facts with a saved on/off
  preference for learners who want additional technical depth;
- adds controllable label noise, three ready-made experiment challenges, and
  separate seeded training and test datasets;
- adds draggable, spring-based physics to the network node graph; and
- includes a freeform Neural Workbench for adding, moving and deleting nodes,
  drawing individual connections, editing weights and biases, training the
  resulting graph and testing it against the shared dataset. A delayed,
  optional string mode lets a learner drag across connections to play them:
  edge colour changes note and timbre, shorter edges play higher pitches,
  longer edges play lower pitches, and stroke speed changes strength.

## Implementation

The network is implemented directly in TypeScript rather than delegated to a
machine-learning service. `app/neural-core.ts` contains deterministic dataset
generation, Xavier-style weight initialisation, forward propagation, binary
cross-entropy evaluation, and batch-gradient backpropagation. The interface in
`app/page.tsx` runs training in short animation-frame batches so the charts and
decision maps remain responsive. The course resets the lab to a small seeded
XOR experiment, then moves from labelled examples and probabilities through
loss, backpropagation, architecture, generalisation and controlled comparison.
Learners can open plain-language, purpose and technical layers separately, and
action segments use their actual clicks, training results and configuration
changes as feedback.

`app/neural-workbench.tsx` contains a second graph-based implementation for the
open editor. It validates the directed graph, evaluates it in topological order
and performs backpropagation over the connections the learner has drawn.

The output layer always uses a sigmoid because the lab performs binary
classification. Hidden layers can use tanh, ReLU, or sigmoid activations.

## Run locally

Requirements: Node.js 22.13 or later.

```bash
npm install
npm run dev
```

Then open the local address printed by the development server.

For the static GitHub Pages version:

```bash
npm run dev:github
```

## Deploy to GitHub Pages

The repository includes `.github/workflows/deploy-pages.yml`. Every push to
`main` builds the browser-only version and publishes `dist-github` through
GitHub Pages. The deployment automatically detects the repository name, so it
works both as a project site such as
`https://writban.github.io/Neural-Network-Visual-Lab/` and as an account site
such as `https://writban.github.io/`.

After the first push, open **Settings → Pages** in GitHub and set **Source** to
**GitHub Actions** if it is not already selected. The Actions tab shows the
deployment run and its resulting URL.

## Project structure

- `app/neural-core.ts`: datasets, model initialisation, inference, metrics, and training
- `app/page.tsx`: experiment state, controls, canvas visualisations, and physics
- `app/neural-workbench.tsx`: editable graph, graph training, testing, and the hidden connection-string instrument
- `app/globals.css`: responsive visual system and component styling
- `github-main.tsx`: static React entry point used by GitHub Pages
- `vite.github.config.ts`: repository-aware static build configuration
- `.github/workflows/deploy-pages.yml`: automatic GitHub Pages deployment

## Design decisions

- **Seeded data and weights:** changing one configuration while retaining the
  seed makes comparisons meaningful and repeatable.
- **Two experiments:** the baseline/challenger layout encourages controlled
  comparisons instead of treating training as a single prepared demonstration.
- **Canvas visualisation:** dense decision surfaces and live loss curves are
  cheaper to redraw on canvas than as hundreds of DOM elements.
- **Editable graph:** the lower workbench uses a directed acyclic graph so
  learners can change individual connections without creating an untrainable
  circular network.
- **No back end:** the datasets are synthetic and the experiments are local, so
  a server would add complexity without improving the learning experience.

## Current limitations

This is an educational binary-classification sandbox, not a production ML
training platform. Computation runs on the browser's main thread, and the
available architectures are deliberately small enough to remain interpretable.
Its train/test comparison is educational: both sets come from the same
synthetic generator, while optional label noise is applied only to training.

Useful future work includes user-drawn datasets, experiment export, a Web
Worker for training, and a TensorFlow.js implementation that can be compared
against the handwritten model.
