from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

rule download_bib_file:
    input:
        HTTP.remote("https://raw.githubusercontent.com/jfcrenshaw/bibfile/main/My%20Library.bib")
    output:
        "src/tex/bib.bib"
    shell:
        "cp {input[0]} {output[0]}"

rule download_bandpasses:
    output:
        directory("src/data/bandpasses")
    script:
        "src/scripts/download_bandpasses.py"

rule plot_decrements:
    output:
        "src/figures/decrements.pdf"
    script:
        "src/scripts/plot_decrements.py"

rule train_ensembles:
    input:
        "src/data/Euclid_trim_27p10_3p5_IR_4NUV.dat"
    output:
        directory("src/data/models")
    cache:
        True
    script:
        "src/scripts/train_ensembles.py"

rule create_observed_catalogs:
    input:
        "src/data/Euclid_trim_27p10_3p5_IR_4NUV.dat"
    output:
        directory("src/data/observed_catalogs")
    script:
        "src/scripts/create_observed_catalogs.py"

rule perform_redshift_cuts:
    input:
        directory("src/data/models"),
        directory("src/data/observed_catalogs"),
        "src/scripts/utils/sample_with_errors.py"
    output:
        directory("src/data/background_catalogs"),
        directory("src/data/foreground_catalogs")
    script:
        "src/scripts/perform_redshift_cuts.py"

rule calculate_photoz_metrics:
    input:
        directory("src/data/observed_catalogs"),
        directory("src/data/background_catalogs"),
        directory("src/data/foreground_catalogs"),
        "src/scripts/utils/survey_areas.py"
    output:
        "src/data/photoz_metrics_bg.pkl",
        "src/data/photoz_metrics_fg.pkl"
    script:
        "src/scripts/calculate_photoz_metrics.py"

rule calculate_sigma_du:
    input:
        directory("src/data/models"),
        directory("src/data/background_catalogs"),
        "src/scripts/utils/sample_with_errors.py"
    output:
        "src/data/sigma_du.pkl"
    script:
        "src/scripts/calculate_sigma_du.py"

rule calculate_correlations:
    input:
        "src/data/Euclid_trim_27p10_3p5_IR_4NUV.dat",
        "src/scripts/utils/load_truth_catalog.py",
        "src/scripts/utils/build_tracer.py",
    output:
        "src/data/correlations.pkl"
    script:
        "src/scripts/calculate_correlations.py"

rule calculate_correlation_snr:
    input:
        "src/data/sigma_du.pkl",
        "src/data/correlations.pkl",
        "src/data/photoz_metrics_bg.pkl",
        "src/scripts/utils/survey_areas.py"
    output:
        "src/data/correlation_snr.pkl"
    script:
        "src/scripts/calculate_correlation_snr.py"

rule save_variables:
    input:
        "src/scripts/utils/sample_with_errors.py",
        "src/data/photoz_metrics_bg.pkl",
        "src/data/photoz_metrics_fg.pkl"
    output:
        "src/tex/output/bg_completeness_y1.txt",
        "src/tex/output/bg_completeness_y10.txt",
        "src/tex/output/bg_completeness_y10+euclid.txt",
        "src/tex/output/bg_completeness_y10+roman.txt",
        "src/tex/output/bg_purity_y1.txt",
        "src/tex/output/bg_purity_y10.txt",
        "src/tex/output/bg_size_y1.txt",
        "src/tex/output/bg_size_y10.txt",
        "src/tex/output/bg_size_y10+euclid+roman.txt",
        "src/tex/output/fg_size_y1.txt",
        "src/tex/output/fg_size_y10.txt",
        "src/tex/output/fg_size_y10+euclid+roman.txt",
        "src/tex/output/m_samples.txt",
        "src/tex/output/zu_samples.txt",
        "src/tex/output/snr_y10.txt",
        "src/tex/output/snr_wff_y1.txt",
        "src/tex/output/snr_wff_y10.txt",
        "src/tex/output/snr_wff_y10+euclid+roman.txt",
        "src/tex/output/snr_wfg_y1.txt",
        "src/tex/output/snr_wfg_y10.txt",
        "src/tex/output/snr_wfg_y10+euclid+roman.txt",
        "src/tex/output/euclid_overlap.txt",
        "src/tex/output/roman_overlap.txt"
    script:
        "src/scripts/save_variables.py"