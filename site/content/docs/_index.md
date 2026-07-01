---
linkTitle: "Documentation index"
title: Introduction
---

👋 Hello! Welcome to the deisa-dask documentation!


## What is deisa-dask?
deisa-dask is a Python library that enables **in situ analytics** for MPI-based high-performance computing (HPC) applications.
It connects distributed simulations to the Dask ecosystem, allowing analytics, visualization, and machine learning workflows to process simulation data directly in memory as it is produced.

deisa-dask exposes distributed simulation data as native Dask arrays.
Existing Dask workflows can therefore be reused with minimal modifications, eliminating the need to write and reload large intermediate datasets from storage.

By combining the scalability of MPI simulations with Dask's flexible task-based programming model, deisa-dask reduces I/O bottlenecks and enables scientists to perform real-time analysis, monitoring, and data reduction on large-scale simulations.


## Features
* **In situ analytics for HPC**
  Analyze simulation data directly in memory without writing intermediate results to disk.
* **MPI and Dask integration**
  Seamlessly connect MPI-based simulations with Dask's distributed computing ecosystem.
* **Minimal code changes**
  Reuse existing Dask analytics workflows with only minor modifications compared to traditional post hoc analysis.
* **Native Dask data structures**
  Access simulation outputs as Dask arrays and futures that integrate naturally with the Dask API.
* **Rich Python ecosystem support**
  Leverage popular libraries such as NumPy, Scikit-Learn, and Dask-ML on live simulation data.
* **High-performance data movement**
  Transfer data directly from simulation processes to Dask workers, avoiding costly file-system I/O.
* **PDI-based instrumentation**
  Integrates with the [PDI Data Interface][PDI], enabling data extraction from simulations with limited intrusion into application code.
* **Scalable distributed execution**
  Designed for large-scale HPC environments where simulations and analytics run across many nodes.
* **Flexible workflow design**
  Supports analytics, visualization, machine learning, monitoring, and data reduction workflows within the same framework.
* **Open and extensible architecture**
  Built using standard Python and Dask components, making it easy to extend and integrate into existing scientific workflows.


## Questions or Feedback?
{{< callout emoji="❓" >}}
  deisa-dask is still in active development.
  Have a question or feedback? Feel free to [open an issue](https://github.com/deisa-project/deisa-dask/issues)!
{{< /callout >}}


## Next
Dive right into the following section to get started:

{{< cards >}}
  {{< card link="getting-started" title="Getting Started" icon="document-text" subtitle="Learn how to use deisa-dask on a simple code example." >}}
{{< /cards >}}

[PDI]: https://pdi.dev

