# Project explanation

This document describes the files in this project and explains the Python
source by line number. Line numbers refer to the current versions of
`phishing_url_detector.py`, `app.py`, and `README.md`; blank lines are only
visual separators. The CSV's data rows and binary image/cache files are
explained by structure and purpose rather than repeated byte-for-byte.

## What the project does

The project trains a Random Forest classifier on URL text features, evaluates
it on a held-out portion of a labeled CSV, and estimates whether new URL text
looks like a phishing URL. It does not open, resolve, fetch, or otherwise send
the analyzed URL anywhere. The command-line program writes `report.json` and
`confusion_features.png`; `app.py` provides an interactive Streamlit interface.

This is a heuristic machine-learning detector, not a browser or website
inspection tool. Its prediction is not proof that a site is safe or malicious.

## `phishing_url_detector.py` — command-line program and shared logic

### Lines 1–29: purpose, dependencies, and constants

- **1–8:** The module docstring names the system, says it uses URL text features
  and Random Forest, and gives the two supported command-line forms. The
  default dataset path is `phish.csv`.
- **9:** Imports Python regular-expression support, used for URL word splitting
  and checking IPv4-like host strings.
- **10:** Imports `sys` to read command-line arguments.
- **11:** Imports `json` to write the analysis report.
- **12:** Imports `math` for the logarithm in Shannon entropy.
- **13:** Imports pandas for CSV loading and feature-table construction.
- **14–16:** Imports Matplotlib, selects its non-interactive `Agg` backend so
  chart creation works without a display, then imports the plotting API.
- **17:** Imports `urlparse` to split URL text into scheme, host, path, query,
  and port without requesting the URL.
- **18:** Imports the Random Forest classifier.
- **19:** Imports the train/test splitting helper.
- **20–21:** Imports the four evaluation metrics and confusion-matrix helper.
- **23–24:** Set the output paths to `report.json` and
  `confusion_features.png`, relative to the process's current working
  directory.
- **26–27:** Define sensitive terms counted in a URL, such as `login`,
  `password`, and `verify`. A term is counted if it occurs anywhere in the
  lowercased URL.
- **28:** Lists recognized URL-shortener hostnames.
- **29:** Lists top-level domain strings treated as suspicious by the simple
  rule-based indicator.

### Lines 32–37: `entropy`

- **32:** Defines `entropy(text)`, a helper that returns a numeric measure of
  character diversity.
- **33:** Documents that the calculation is Shannon entropy.
- **34–35:** Returns `0.0` for an empty string, avoiding a division by zero.
- **36:** Calculates each distinct character's share of the string.
- **37:** Returns `-sum(p * log2(p))`, the Shannon entropy formula. A more varied
  character distribution generally has higher entropy.

### Lines 40–83: `extract_features`

- **40–41:** Defines the URL-to-feature function and states that it uses lexical
  features only.
- **42:** Removes whitespace at the start and end of the supplied URL.
- **43–44:** Tries to parse the input. If it has no `://`, it prepends
  `http://` for parsing; this does not contact a server.
- **45:** Reads the parsed hostname, substitutes an empty string if absent, and
  normalizes it to lowercase.
- **46:** Reads the parsed port. Malformed port values can raise `ValueError`.
- **47–48:** On a parsing `ValueError`, substitutes an inert placeholder parse
  and clears the host and port.
- **49:** Gets path and query text, using empty strings if either is missing.
- **50:** Splits the hostname on dots and removes empty labels.
- **51:** Splits the original trimmed URL into non-word chunks; underscores
  count as separators for this calculation.
- **52:** Creates a lowercase copy used for case-insensitive keyword checks.
- **53:** Begins the dictionary of 29 features returned to pandas and the model.
- **54:** `url_length` is the number of characters in the trimmed URL.
- **55:** `host_length` is the hostname's character count.
- **56:** `path_length` is the parsed path's character count.
- **57:** `query_length` is the parsed query's character count.
- **58:** `num_dots` counts periods in the entire URL.
- **59:** `num_hyphens` counts hyphens in the entire URL.
- **60:** `num_underscores` counts underscores in the entire URL.
- **61:** `num_slashes` counts slashes in the entire URL.
- **62:** `num_digits` counts digit characters in the entire URL.
- **63:** `digit_ratio` divides the digit count by URL length; `max(..., 1)`
  prevents division by zero.
- **64–68:** Count, respectively, `@`, `?`, `=`, `&`, and `%` characters in
  the entire URL.
- **69:** `has_ip` is 1 if the host text matches four one-to-three-digit groups
  separated by dots, otherwise 0. This is a shape check, not strict validation
  that each octet is in the IPv4 range.
- **70:** `is_https` is 1 only when the parsed scheme is exactly `https`.
- **71:** Estimates subdomains by counting hostname labels minus two, with a
  floor of zero. It is a simple approximation, not public-suffix-aware.
- **72:** Counts hyphens in the hostname.
- **73:** Counts digits in the hostname.
- **74:** Counts how many configured sensitive terms occur in the lowercase
  URL. Each listed term contributes at most one, regardless of repetitions.
- **75:** `is_shortener` is 1 when the entire hostname exactly equals one of the
  configured shortener hosts.
- **76:** `suspicious_tld` is 1 if the last hostname label is in the configured
  suspicious-TLD list.
- **77:** `has_port` is 1 when parsing found an explicit port.
- **78:** `has_punycode` is 1 when the hostname contains `xn--`.
- **79:** `http_in_path` is 1 if the path or query contains `http`, ignoring
  letter case.
- **80:** `starts_with_www` is 1 if the normalized hostname begins with `www.`.
- **81:** `longest_word` is the maximum length of the URL chunks from line 51;
  it is zero if there are no chunks.
- **82:** `url_entropy` applies the entropy helper to the trimmed URL text.
- **83:** Ends and returns the feature dictionary.

### Lines 86–107: `indicators`

- **86–87:** Defines a helper that turns selected features into readable
  warning strings. The `url` parameter is accepted but not used in the body;
  decisions are based on `f`, the feature dictionary.
- **88:** Starts with an empty list of flags.
- **89–90:** Adds a warning when the host has the IPv4-like shape described
  above.
- **91–92:** Adds a warning when the URL contains `@`.
- **93–94:** Adds a warning when the scheme was not HTTPS.
- **95–96:** Adds a warning when the hostname exactly matches a listed
  shortening service.
- **97–98:** Adds a warning for at least three estimated subdomains.
- **99–100:** Adds a warning when at least two configured keywords occur.
- **101–102:** Adds a warning when the final hostname label is on the
  suspicious-TLD list.
- **103–104:** Adds a warning when URL length is greater than 75 characters.
- **105–106:** Adds a warning when the hostname contains punycode notation.
- **107:** Returns all triggered warnings. These rules supplement, but do not
  determine, the model's classification.

### Lines 110–131: loading, training, and evaluation

- **110:** Defines `load_data(path)`.
- **111:** Reads the CSV, selects only the `url` and `status` columns, and
  removes rows where either selected value is missing. All other CSV columns
  are ignored.
- **112:** Extracts the 29 URL-text features for every remaining URL and builds
  a pandas feature table.
- **113:** Converts labels to integers: exactly `phishing` becomes 1; any other
  non-missing status becomes 0 (legitimate).
- **114:** Returns the feature table `X` and target labels `y`.
- **117:** Defines training/evaluation and defaults to `phish.csv`.
- **118:** Loads the selected CSV and extracts its features and labels.
- **119–120:** Makes a stratified 70/30 train/test split. `random_state=42`
  makes the split repeatable for the same data and library behavior;
  `stratify=y` preserves the class proportions in both portions.
- **121:** Creates a 200-tree Random Forest with a fixed seed and uses all
  available CPU cores (`n_jobs=-1`).
- **122:** Fits the model to the training portion.
- **123:** Predicts class labels for the held-out test portion.
- **124:** Starts a dictionary of evaluation results.
- **125:** Accuracy is the fraction of test predictions that are correct.
- **126:** Precision is the fraction of predicted phishing URLs that are
  actually phishing.
- **127:** Recall is the fraction of actual phishing URLs found by the model.
- **128:** F1 is the harmonic mean of precision and recall.
- **129:** Creates the confusion matrix and converts it to nested Python lists
  so it can be serialized as JSON. With the default label ordering, rows are
  actual classes and columns are predicted classes.
- **130:** Closes the metrics dictionary.
- **131:** Returns the trained model, feature names in their training order,
  metric values, and the train/test row counts.

### Lines 134–142: analyzing one URL

- **134:** Defines the function used by both the command-line program and web
  app.
- **135:** Extracts the same 29 features used during training.
- **136:** Makes a one-row feature table in the saved column order, gets the
  model's class probabilities, selects the phishing class (index 1), and
  converts it to a built-in `float`.
- **137:** Maps probability to risk: HIGH at 0.8 or above, MEDIUM from 0.5
  through below 0.8, and LOW below 0.5.
- **138:** Begins the result dictionary.
- **139:** Classifies a URL as phishing at probability 0.5 or above; otherwise
  it is legitimate.
- **140:** Rounds the reported phishing probability to three decimal places.
- **141:** Stores the risk label.
- **142:** Stores the independently generated rule-based indicators and ends
  the result.

### Lines 145–165: creating the chart

- **145–146:** Defines chart generation and describes its two panels.
- **147:** Creates a side-by-side figure sized 13 by 4.5 inches.
- **148:** Displays the confusion matrix using a blue color scale.
- **149:** Sets the left-panel title.
- **150–151:** Labels the horizontal and vertical matrix axes with the two
  classes.
- **152–153:** Names the horizontal axis `Predicted` and vertical axis `Actual`.
- **154:** Finds the largest cell value, used to choose readable text colors.
- **155–156:** Loops over the two matrix rows and columns.
- **157–158:** Writes each cell count in its center; uses white text for darker
  cells and black text for lighter cells.
- **159:** Pairs the model's feature importance values with their names, selects
  the top ten, then reverses the order for horizontal-bar plotting.
- **160:** Draws the horizontal feature-importance bars.
- **161–162:** Sets the second panel's title and x-axis label.
- **163:** Adjusts spacing so labels and titles fit.
- **164:** Saves the chart to `confusion_features.png` at 150 DPI.
- **165:** Closes the figure to release plotting resources.

### Lines 168–201: command-line entry point

- **168:** Defines `main`; the optional `argv` parameter makes it possible to
  pass arguments directly when calling this function in code.
- **169:** Uses process command-line arguments after the program name, unless
  the caller provided `argv`.
- **170:** Uses the first argument as the dataset path or defaults to
  `phish.csv`.
- **171:** Trains/evaluates the model and unpacks the row counts.
- **172:** Prints training count, testing count, and number of features.
- **173–176:** Print accuracy, precision, recall, and F1 to four decimal places.
- **177:** Prints the confusion matrix and identifies the cell ordering as
  `[[TN, FP], [FN, TP]]`.
- **179–185:** Defines five fixed demonstration URLs: a normal-looking
  Wikipedia URL, an IP-host URL, a suspicious-looking long URL, a Python
  documentation URL, and a shortener URL. They are only strings; no URL is
  visited.
- **186:** Analyzes each demonstration URL with the trained model.
- **187:** Prints a heading for the URL findings.
- **188–190:** Loops over findings and prints risk, prediction, probability,
  and at most the first 60 characters of each URL.
- **192–193:** Assembles a report containing the model description, full
  dataset row count, evaluation metrics, and demonstration URL results.
- **194:** Opens (or creates) `report.json` for writing, replacing an earlier
  report in the current working directory.
- **195:** Writes the report as indented JSON.
- **196:** Generates and saves the chart.
- **197:** Prints a completion message naming the output files.
- **200:** Runs the entry-point check. Importing this module from `app.py` does
  not run the command-line workflow.
- **201:** Calls `main()` when this file itself is executed as a script.

## `app.py` — Streamlit dashboard

Line numbers refer to this file.

- **1:** Module docstring describing the web interface.
- **2–3:** Import JSON serialization and `Path` for file paths.
- **5–6:** Import pandas for display tables and Streamlit for the UI.
- **8:** Reuse the detector's URL analysis and model-training functions instead
  of implementing a second model.
- **11:** Resolve the default CSV beside `app.py`, not relative to the shell's
  current directory.
- **13–17:** Configure the page title, shield icon, and wide layout. The icon
  is an emoji; its displayed encoding may depend on the source-file/editor
  encoding.
- **20–22:** Cache the loaded/trained model as a Streamlit resource. Dataset
  path, modification time, and file size are function arguments so changing
  the selected file or its metadata causes Streamlit to compute a new cache
  entry. The function itself delegates to `train_and_evaluate`.
- **25–29:** Show the page title and explain that scanning uses text only.
- **31–38:** Build a sidebar with the dataset path field and a radio control
  for the Model overview, URL scanner, and Project details pages.
- **40:** Expand a leading `~` and resolve the selected path to an absolute path.
- **41–43:** If the path is not a file, show an error and stop rendering the
  app for that run.
- **45:** Read file metadata used by the cache key.
- **46–48:** Load or reuse the trained model, feature columns, metrics, and
  train/test counts.
- **50–56:** For Model overview, show the four test-set metrics as percentages.
- **58–61:** Show model type, tree count, data split sizes, and feature count.
- **62–70:** Create two columns and show the confusion matrix as a labeled
  dataframe.
- **71–82:** Show a horizontal bar chart of the ten highest model feature
  importances.
- **84–89:** Start the URL scanner view and explain that pasted links are not
  visited, resolved, or sent to a third party.
- **90–99:** Put a multiline URL input and submit button inside a form. Example
  text illustrates one URL per line.
- **101:** Process input only after the form is submitted.
- **102:** Split input into lines, trim whitespace, and discard blank lines.
- **103–104:** Warn if the submitted form contains no URLs.
- **105–106:** Otherwise analyze each URL using the already-loaded model.
- **107–120:** Build and display a summary table with URL, title-cased
  prediction, formatted phishing probability, and risk level.
- **122–123:** Iterate over results and create one expandable detail section
  per URL.
- **124–129:** Display the prediction, probability, and risk level in that
  section.
- **130–135:** List triggered rule-based indicators, or report that none were
  found.
- **137–142:** Assemble a downloadable report with model name, dataset size,
  metrics, and URL results.
- **143–148:** Add a button that provides the report as formatted JSON with an
  application/json content type.
- **150–166:** The remaining navigation choice is Project details. It explains
  the five pipeline steps, risk thresholds, and limitations; the displayed
  content is a Markdown string rather than executable analysis logic.

## `README.md` — setup and usage

- **1:** Project title.
- **3:** Short description and the important no-URL-fetching design statement.
- **5–9:** Setup heading and a pip command listing pandas, numpy,
  scikit-learn, matplotlib, and Streamlit.
- **11–13:** Dataset acquisition instructions. The CSV should be saved beside
  the script as `phish.csv` and must include `url` and `status` columns.
- **15–21:** Run commands for the default CLI dataset, an alternate CSV, and
  the Streamlit dashboard.
- **23–28:** Lists console output, generated JSON and chart, and dashboard
  features.
- **30–33:** Summarizes the model pipeline and risk thresholds.
- **35:** Reiterates that analysis is local and based on URL text alone.

## `phish.csv` — labeled input data

This is a CSV with **11,430 data rows** plus a header, and the present file is
balanced: 5,715 rows labeled `legitimate` and 5,715 labeled `phishing`.
`load_data` uses only `url` and `status`, drops rows missing either, extracts
its own 29 features from `url`, and derives the binary target from `status`.
The other precomputed columns are not used by this implementation.

The header columns, grouped by their names and apparent role:

- **URL and basic lexical fields:** `url`, `length_url`, `length_hostname`,
  `ip`, `nb_dots`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_and`, `nb_or`,
  `nb_eq`, `nb_underscore`, `nb_tilde`, `nb_percent`, `nb_slash`, `nb_star`,
  `nb_colon`, `nb_comma`, `nb_semicolumn`, `nb_dollar`, `nb_space`,
  `nb_www`, `nb_com`, `nb_dslash`.
- **URL syntax, host, and word fields:** `http_in_path`, `https_token`,
  `ratio_digits_url`, `ratio_digits_host`, `punycode`, `port`, `tld_in_path`,
  `tld_in_subdomain`, `abnormal_subdomain`, `nb_subdomains`, `prefix_suffix`,
  `random_domain`, `shortening_service`, `path_extension`,
  `nb_redirection`, `nb_external_redirection`, `length_words_raw`,
  `char_repeat`, `shortest_words_raw`, `shortest_word_host`,
  `shortest_word_path`, `longest_words_raw`, `longest_word_host`,
  `longest_word_path`, `avg_words_raw`, `avg_word_host`, `avg_word_path`.
- **Brand and phishing hints:** `phish_hints`, `domain_in_brand`,
  `brand_in_subdomain`, `brand_in_path`, `suspecious_tld`,
  `statistical_report`.
- **Page/link and form fields:** `nb_hyperlinks`, `ratio_intHyperlinks`,
  `ratio_extHyperlinks`, `ratio_nullHyperlinks`, `nb_extCSS`,
  `ratio_intRedirection`, `ratio_extRedirection`, `ratio_intErrors`,
  `ratio_extErrors`, `login_form`, `external_favicon`, `links_in_tags`,
  `submit_email`, `ratio_intMedia`, `ratio_extMedia`, `sfh`, `iframe`,
  `popup_window`, `safe_anchor`, `onmouseover`, `right_clic`, `empty_title`,
  `domain_in_title`, `domain_with_copyright`.
- **Domain/reputation and label fields:** `whois_registered_domain`,
  `domain_registration_length`, `domain_age`, `web_traffic`, `dns_record`,
  `google_index`, `page_rank`, `status`.

The dataset may include page/domain-derived attributes, but this project does
not inspect pages or perform those lookups when it analyzes a new URL.

## Generated files and cache

- **`report.json`:** A generated snapshot containing the model description,
  dataset size, four test-set metrics, confusion matrix, and the five sample
  URL findings from the CLI run. Its numbers are results from one run, not
  guaranteed values for future data or environments. The CLI overwrites it
  when rerun. In the copy currently in the project, `dataset_size` is 11,430;
  accuracy is about 0.8845, precision 0.8774, recall 0.8938, and F1 0.8855.
  The confusion matrix is `[[1501, 214], [182, 1532]]`, in
  `[[true negative, false positive], [false negative, true positive]]` order.
  Each entry in `url_reports` has the original URL, predicted class,
  phishing probability, risk level, and the list of triggered indicators.
  The five entries are outputs for the five demonstration URLs in the CLI
  source; they are not labels supplied by the CSV.
- **`confusion_features.png`:** A generated image with the held-out confusion
  matrix and ten highest feature importances. The matrix compares actual
  classes on its vertical axis with predicted classes on its horizontal axis;
  the bar chart ranks the features the fitted forest used most. It is
  overwritten by the CLI.
- **`__pycache__/phishing_url_detector.cpython-312.pyc`:** Python's generated
  bytecode cache for Python 3.12. It is not hand-maintained source and can be
  regenerated by Python.
- **`.git/`:** Git's internal repository metadata, not an application file.

