# Legacy MDS artifacts

These 1,215 plots and raw CSV files were produced by the pre-1.0 MDS path.
That implementation converted only explicitly stored sparse affinities, so an
implicit zero affinity incorrectly became zero distance instead of maximal
distance. They are retained for provenance and must not be used as current
scientific results. New MDS runs use
`forestgap.applications.visualization.plot_mds`, which routes through the shared
`prepare_proximity_affinity` and `proximity_to_distance` conversion.
