@app.route("/upload_dataset", methods=["POST"])
@login_required(role="teacher")
def upload_dataset():
    """
    Teacher uploads CSV/XLSX dataset.
    Save original, store df in memory, also save as latest_dataset.csv,
    train models and generate charts.
    """
    global current_df, best_model_name, features_global, scores_global, feature_importances_global

    try:
        best_model_name = None
        features_global = []
        scores_global = {}
        feature_importances_global = None

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file field in form"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            return jsonify(
                {
                    "success": False,
                    "error": "Only CSV or Excel files are allowed (.csv, .xlsx, .xls)",
                }
            ), 400

        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        if ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        if df is None or df.empty:
            return jsonify(
                {"success": False, "error": "File has no data or could not be read"}
            ), 400

        current_df = df.copy()

        # persist dataset for dashboard reload
        latest_path = os.path.join(DATA_FOLDER, "latest_dataset.csv")
        df.to_csv(latest_path, index=False)

        scores = train_models(current_df)
        summary = build_dataset_summary(current_df)

        return jsonify({"success": True, "scores": scores, "summary": summary})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": Fal