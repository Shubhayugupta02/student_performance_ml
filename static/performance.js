document.addEventListener("DOMContentLoaded", () => {
  const errorEl = document.getElementById("perfError");

  // Small helper to safely get canvas (handle missing elements)
  function getCtx(id) {
    const el = document.getElementById(id);
    return el ? el.getContext("2d") : null;
  }

  async function loadDashboard() {
    try {
      const res = await fetch("/api/dashboard");
      const data = await res.json();

      if (!res.ok || data.error) {
        errorEl.textContent = data.error || "Failed to load dashboard data.";
        console.error("Dashboard API error:", data);
        return;
      }

      errorEl.textContent = "";

      // ---------- Grade Distribution ----------
      const grades = data.grades || {};
      const gradeLabels = Object.keys(grades);
      const gradeValues = gradeLabels.map(k => grades[k]);

      const ctxGrades = getCtx("chartGrades");
      if (ctxGrades) {
        new Chart(ctxGrades, {
          type: "pie",
          data: {
            labels: gradeLabels,
            datasets: [{
              data: gradeValues,
            }]
          },
          options: {
            plugins: {
              legend: { position: "bottom", labels: { color: "#e5e7eb" } },
            }
          }
        });
      }

      // ---------- Helper to build bar chart ----------
      function barChart(ctx, students, title) {
        if (!ctx || !students || !students.length) return;
        const labels = students.map(s => s.name);
        const vals = students.map(s => s.final);
        new Chart(ctx, {
          type: "bar",
          data: {
            labels,
            datasets: [{
              label: title,
              data: vals,
            }]
          },
          options: {
            indexAxis: "y",
            scales: {
              x: {
                beginAtZero: true,
                ticks: { color: "#e5e7eb" }
              },
              y: {
                ticks: { color: "#e5e7eb" }
              }
            },
            plugins: {
              legend: { display: false },
              title: {
                display: false
              }
            }
          }
        });
      }

      // ---------- Top & Bottom charts ----------
      barChart(getCtx("chartTop10"), data.top10, "Top 10");
      barChart(getCtx("chartBottom10"), data.bottom10, "Bottom 10");

      // ---------- Weak students list ----------
      const weakListEl = document.getElementById("weakList");
      if (weakListEl) {
        weakListEl.innerHTML = "";
        (data.weak_students || []).forEach((s) => {
          const li = document.createElement("li");
          li.textContent = `${s.name}: ${s.final.toFixed(2)}%`;
          weakListEl.appendChild(li);
        });
        if (!weakListEl.children.length) {
          weakListEl.innerHTML =
            "<li>No students below 40% in this dataset.</li>";
        }
      }

      // ---------- Feature importance ----------
      const importance = data.importance || {};
      const impLabels = Object.keys(importance);
      const impVals = impLabels.map(k => importance[k]);

      const ctxImp = getCtx("chartImportance");
      if (ctxImp && impLabels.length) {
        new Chart(ctxImp, {
          type: "bar",
          data: {
            labels: impLabels,
            datasets: [{
              label: "Importance",
              data: impVals
            }]
          },
          options: {
            scales: {
              x: { ticks: { color: "#e5e7eb" } },
              y: { beginAtZero: true, ticks: { color: "#e5e7eb" } }
            },
            plugins: {
              legend: { display: false }
            }
          }
        });
      } else if (ctxImp) {
        // show a text when no importance
        ctxImp.canvas.parentElement.innerHTML =
          "<p style='font-size:12px;'>Feature importance is only available when Random Forest is selected as the best model.</p>";
      }
    } catch (err) {
      console.error("Dashboard fetch failed:", err);
      errorEl.textContent = "Error contacting server.";
    }
  }

  loadDashboard();
});



