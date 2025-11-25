// ---- Dataset upload ----
document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("uploadForm");
  const datasetFile = document.getElementById("datasetFile");
  const uploadStatus = document.getElementById("uploadStatus");
  const uploadSummary = document.getElementById("uploadSummary");
  const summaryText = document.getElementById("summaryText");
  const scoreText = document.getElementById("scoreText");

  if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = datasetFile.files[0];
      if (!file) {
        alert("Please choose a file.");
        return;
      }
      const fd = new FormData();
      fd.append("file", file);

      uploadStatus.textContent = "Uploading & training model...";
      uploadSummary.classList.add("hidden");

      try {
        const res = await fetch("/upload_dataset", {
          method: "POST",
          body: fd,
        });
        const data = await res.json();
        if (!data.success) {
          uploadStatus.textContent = "Error: " + (data.error || "unknown");
          return;
        }
        uploadStatus.textContent = "Model trained successfully.";
        const summary = data.summary;
        summaryText.textContent =
          `Rows: ${summary.rows}, Columns: ${summary.cols}`;
        const scores = data.scores;
        scoreText.innerHTML =
          `Linear: ${(scores["Linear Regression"] * 100).toFixed(2)}% &nbsp; ` +
          `Tree: ${(scores["Decision Tree"] * 100).toFixed(2)}% &nbsp; ` +
          `RF: ${(scores["Random Forest"] * 100).toFixed(2)}%`;
        uploadSummary.classList.remove("hidden");
      } catch (err) {
        console.error(err);
        uploadStatus.textContent = "Error uploading dataset.";
      }
    });
  }

  // ---- Prediction ----
  const predictionForm = document.getElementById("predictionForm");
  const resultCard = document.getElementById("resultCard");

  if (predictionForm) {
    predictionForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const prevPercent = document.getElementById("prevPercent").value || 0;
      const internalMarks = document.getElementById("internalMarks").value || 0;
      const attendance = document.getElementById("attendance").value || 0;
      const studyHours = document.getElementById("studyHours").value || 0;
      const assignments = document.getElementById("assignments").value || 0;

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prev_percent: prevPercent,
            internal_marks: internalMarks,
            attendance: attendance,
            study_hours: studyHours,
            assignments: assignments,
          }),
        });
        const data = await res.json();
        if (data.error) {
          alert(data.error);
          return;
        }

        document.getElementById("predictedVal").textContent =
          data.prediction + " %";
        document.getElementById("predictedGrade").textContent = data.grade;
        document.getElementById("percentile").textContent = data.percentile;
        document.getElementById("recommendationText").textContent =
          data.recommendation;
        document.getElementById("modelName").textContent =
          data.best_model || "";
        document.getElementById("modelScore").textContent =
          data.best_score || 0;
        document.getElementById("riskText").textContent =
          data.risk ? "AT RISK" : "Safe / Normal";

        resultCard.classList.remove("hidden");

        // attach data for PDF button
        resultCard.dataset.payload = JSON.stringify(data);
      } catch (err) {
        console.error(err);
        alert("Error calling prediction API.");
      }
    });
  }

  // ---- PDF Report ----
  const pdfBtn = document.getElementById("downloadPdfBtn");
  if (pdfBtn) {
    pdfBtn.addEventListener("click", async () => {
      const card = document.getElementById("resultCard");
      if (!card || !card.dataset.payload) return;

      const baseData = JSON.parse(card.dataset.payload);
      const name = document.getElementById("studentName").value || "Student";

      const body = {
        student_name: name,
        prev_percent: document.getElementById("prevPercent").value || 0,
        internal_marks: document.getElementById("internalMarks").value || 0,
        attendance: document.getElementById("attendance").value || 0,
        study_hours: document.getElementById("studyHours").value || 0,
        assignments: document.getElementById("assignments").value || 0,
        prediction: baseData.prediction,
        grade: baseData.grade,
        percentile: baseData.percentile,
        risk: baseData.risk,
        recommendation: baseData.recommendation,
        best_model: baseData.best_model,
        best_score: baseData.best_score,
      };

      try {
        const res = await fetch("/download_report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          alert("Failed to generate PDF.");
          return;
        }
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "student_report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        console.error(err);
        alert("Error downloading PDF.");
      }
    });
  }
});






