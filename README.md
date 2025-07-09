## Seminar: Quality in human and machine translation (LST, UdS, SoSe-2025)
### Seminar companion

Code used to setup a small manual effort to annotate MT quality and calculated five automatic metrics.

MT was generated via DeepL web interface (30 June 2025).

Data: two documents for each translation direction (DEEN and ENDE) from Europarl UdS v2 corpus, 
identified as most challenging based on a number of translation task difficulty indicators. 
The documents are selected from a larger pool of difficult documents as having the highest and the lowest sum-aggregate rank calculated from the ranks assigned using the five automatic metrics. 

The total number of segments to annotate is 33 and 36 for DEEN and ENDE, respectively.

The expected number of raters per translation direction: 3

### Content
- calculate automatic quality scores: `automatic_metrics.py`
- design and format the annotation spreadsheet, inc. selecting the most challenging docs: `manual_setup.py`
- fill in dummy/proxy annotations and collect scores from real annotators: `anno_postpro.py`
- count and visualise interrater reliability: `interrater_agreement.py`
- correlate manual and automatic scores: `raters_vs_metrics.py`
