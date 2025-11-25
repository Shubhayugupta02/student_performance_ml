async function loadDashboardCharts() {
  const errorEl = document.getElementById("perfError");
  if (!errorEl) return;

  try {
    const res = await fetch("/api/dashboard");
    const data = await res.json();
    if (!res.ok || data.error) {
      errorEl.textContent = data.error || "Error loading dashboard.";
      return;
    }

    const ctxGrades = document.getElementById("gradeChart");
    if (ctxGrades) {
      new Chart(ctxGrades.getContext("2d"), {
        type: "pie",
        data: {
          labels: ["A", "B", "C", "D", "F"],
          datasets: [
            {
              data: [
                data.grades.A,
                data.grades.B,
                data.grades.C,
                data.grades.D,
                data.grades.F,
              ],
            },
          ],
        },
        options: { plugins: { legend: { position: "bottom" } } },
      });
    }

    const makeBar = (canvasId, list, title) => {
      const c = document.getElementById(canvasId);
      if (!c) return;
      new Chart(c.getContext("2d"), {
        type: "bar",
        data: {
          labels: list.map((x) => x.name),
          datasets: [
            {
              data: list.map((x) => x.final),
            },
          ],
        },
        options: {
          plugins: { legend: { display: false }, title: { display: false } },
          scales: {
            x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 30 } },
            y: { title: { display: true, text: data.target } },
          },
        },
      });
    };

    makeBar("top10Chart", data.top10, "Top 10");
    makeBar("bottom10Chart", data.bottom10, "Bottom 10");
    makeBar("weakChart", data.weak_students, "Weak");

    const ctxImp = document.getElementById("importanceChart");
    if (ctxImp && data.importance) {
      const labels = Object.keys(data.importance);
      const vals = Object.values(data.importance);
      new Chart(ctxImp.getContext("2d"), {
        type: "bar",
        data: {
          labels,
          datasets: [{ data: vals }],
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            y: {
              title: { display: true, text: "Importance" },
            },
          },
        },
      });
    }
  } catch (err) {
    console.error(err);
    errorEl.textContent = "Error connecting to dashboard API.";
  }
}

