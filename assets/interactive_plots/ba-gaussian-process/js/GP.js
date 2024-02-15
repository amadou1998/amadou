// --------------------
// GP Chart
// --------------------

function makeGPChart(ctx) {
  Chart.pluginService.register({
    beforeDraw: function (chart, easing) {
      if (
        chart.config.options.chartArea &&
        chart.config.options.chartArea.backgroundColor
      ) {
        var helpers = Chart.helpers;
        var ctx = chart.chart.ctx;
        var chartArea = chart.chartArea;

        ctx.save();
        ctx.fillStyle = chart.config.options.chartArea.backgroundColor;
        ctx.fillRect(
          chartArea.left,
          chartArea.top,
          chartArea.right - chartArea.left,
          chartArea.bottom - chartArea.top
        );
        ctx.restore();
      }
    },
  });

  var gpChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Observations",
          data: [],
          pointStyle: "circle",
          radius: 5,
          borderColor: "#00BFFF",
          backgroundColor: "rgba(0, 191, 255, 0.4)",
          fill: false,
          showLine: false,
          borderRadius: 3,
        },
        {
          label: "Mean",
          data: [],
          pointStyle: "None",
          radius: 0,
          borderColor: "#708090",
          backgroundColor: "#708090",
          fill: false,
          showLine: true,
          linewidth: 3,
        },
        {
          label: "Uncertainity (68%)",
          data: [],
          pointStyle: "None",
          radius: 0,
          borderColor: "rgba(135, 206, 235, 0.7)",
          fill: true,
          showLine: true,
          backgroundColor: "rgba(135, 206, 235, 0.3)",
        },
      ],
    },
    options: {
      plugins: {
          legend: {
              labels: {
                  font: {
                      size: 20,
                      family: "'Times New Roman', Times, serif", 
                  }
              }
          }
      },
      chartArea: {
        backgroundColor: "rgba(247, 247, 247, 1)",
      },
      tooltips: {
        enabled: false,
      },
      hover: { mode: null },
      legend: {
        position: "bottom",
        labels: {
          usePointStyle: true
        },
      },
      scales: {
        xAxes: [
          {
            type: "linear",
            position: "bottom",
            gridLines: {
              drawBorder: true,
              display: true,
              drawOnChartArea: false,
            },
            ticks: {
              display: false,
              min: -5,
              max: 5,
            },
          },
          {
            position: "top",
            ticks: {
              display: false,
            },
            gridLines: {
              display: true,
              drawTicks: false,
              drawOnChartArea: false,
            },
          },
        ],
        yAxes: [
          {
            position: "left",
            gridLines: {
              display: true,
              drawOnChartArea: false,
            },
            ticks: {
              min: -20,
              max: 20,
            },
          },
          {
            position: "right",
            ticks: {
              display: false,
            },
            gridLines: {
              display: true,
              drawTicks: false,
              drawOnChartArea: false,
            },
          },
        ],
      },
    },
  });

  return gpChart;
}

function resetObservations() {
  observations = [[], []];
  calculateGP(activeKernels, x_s);
  replaceData(gpChart, 0, [], []);
}
function addData(chart, x, y) {
  var datapoint = { x: x, y: y };
  chart.data.datasets[0].data.push(datapoint);
  chart.update();
}
function replaceData(chart, idx, xs, ys) {
  var data = [];
  xs.forEach(function (value, index, matrix) {
    data.push({ x: value, y: ys.get(index) });
  });
  chart.data.datasets[idx].data = data;
  chart.update();
}

// --------------------
// Mouse Events
// --------------------

function getCursorPosition(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return [x, y];
}
function getCursorCoordinates(canvas, myChart, event) {
  var ytop = myChart.chartArea.top;
  var ybottom = myChart.chartArea.bottom;
  var ymin = myChart.scales["y-axis-0"].min;
  var ymax = myChart.scales["y-axis-0"].max;

  var xleft = myChart.chartArea.left;
  var xright = myChart.chartArea.right;
  var xmin = myChart.scales["x-axis-0"].min;
  var xmax = myChart.scales["x-axis-0"].max;

  var clickpos = getCursorPosition(canvas, event);
  var x = clickpos[0];
  var y = clickpos[1];

  if (x < xright && x > xleft && y < ybottom && y > ytop) {
    var xproportion = (x - xleft) / (xright - xleft);
    var xcoord = xproportion * (xmax - xmin) + xmin;

    var yproportion = -(y - ybottom) / (ybottom - ytop);
    var ycoord = yproportion * (ymax - ymin) + ymin;
  }

  return [xcoord, ycoord];
}
function addDataPointAtCursor(canvas, myChart, e) {
  var coords = getCursorCoordinates(canvas, gpChart, e);
  addData(myChart, coords[0], coords[1]);
  observations[0].push(coords[0]);
  observations[1].push(coords[1]);
}



// --------------------
// Mathematical Functions
// --------------------

function m(xs) {
  return math.zeros(len(xs));
}
function linspace(low, high, n) {
  var step = (high - low) / (n - 1);
  return math.range(low, high, step, true);
}
function flip(matrix) {
  var idx = math.range(0, len(matrix));
  var flippedidx = math.subtract(idx.get([len(idx) - 1]), idx);
  return matrix.subset(math.index(flippedidx));
}
function len(matrix, axis = 0) {
  if (matrix instanceof Array) {
    length = matrix.length;
  } else {
    length = matrix.size()[axis];
  }
  return length;
}
function pairwise_diffenerence(matrix1, matrix2) {
  pd = math.zeros(len(matrix1), len(matrix2));
  matrix1.forEach(function (m1, idx1) {
    matrix2.forEach(function (m2, idx2) {
      pd._data[idx1][idx2] = m1 - m2;
    });
  });
  return pd;
}

// --------------------
// Gaussian Process Implementation
// --------------------

class RBF {
  constructor(sigma, l, n_example = 25) {
    this.sigma = sigma;
    this.l = l;
    this.n_example = n_example;
    this.example_points = linspace(-5, 5, n_example);
  }
  calculate(xs, ys) {
    xs = math.matrix(xs);
    ys = math.matrix(ys);
    var d = pairwise_diffenerence(xs, ys);
    var dl = math.divide(math.square(d), math.square(this.l));
    var e = math.exp(math.multiply(dl, -0.5));
    return math.multiply(math.square(this.sigma), e);
  }
  updateSigma(value) {
    this.sigma = value;
  }
  updateL(value) {
    this.l = value;
  }
  getVisualization() {
    return this.calculate(this.example_points, this.example_points);
  }
}
class ActiveKernels {
  constructor(kernels, method) {
    this.kernels = kernels;
    this.method = method;
  }
  calculate(xs, ys) {
    var results = this.kernels[0].calculate(xs, ys);
    var i;
    for (i = 1; i < this.kernels.length; i++) {
      if (method == "add") {
        results = math.add(results, this.kernels[i].calculate(xs, ys));
      } else {
        results = math.multiply(results, this.kernels[i].calculate(xs, ys));
      }
    }
    return results;
  }
}
function calculateGP(kernel, x_s) {
  var x_obs = observations[0];
  var y_obs = observations[1];

  if (len(observations[0]) == 0) {
    std = math.multiply(kernel.kernels[0].sigma, math.ones(len(x_s)));
    mu_s = m(x_s);
  } else {
    // Calculate kernel components
    var K = kernel.calculate(x_obs, x_obs);
    // Measurement noise
    var sigma_noise = 0;
    var identity = math.identity(K.size());
    var noise = math.multiply(math.square(sigma_noise), identity);
    var K_s = kernel.calculate(x_obs, x_s);
    var K_ss = kernel.calculate(x_s, x_s);
    var K_sTKinv = math.multiply(
      math.transpose(K_s),
      math.inv(math.add(K, noise))
    );
    // New mean
    var mu_s = math.add(
      m(x_s),
      math.squeeze(math.multiply(K_sTKinv, math.subtract(y_obs, m(x_obs))))
    );
    var Sigma_s = math.subtract(K_ss, math.multiply(K_sTKinv, K_s));
    // New std
    var std = math.sqrt(Sigma_s.diagonal());
  }
  var uncertainty = math.multiply(2, std);
  replaceData(gpChart, 1, x_s, mu_s);
  x_s = math.concat(x_s, flip(x_s));
  y_s = math.concat(
    math.add(mu_s, uncertainty),
    flip(math.subtract(mu_s, uncertainty))
  );
  replaceData(gpChart, 2, x_s, y_s);
}
function makexPoints(n) {
  var xmin = gpChart.scales["x-axis-0"].min;
  var xmax = gpChart.scales["x-axis-0"].max;
  return linspace(xmin, xmax + 1, n);
}

// --------------------
// Slider Events
// --------------------

function updateFromSlider(slider, updateAtrFunc, kernel) {
  slider.oninput = function () {
    updateAtrFunc(this.value);
    // Update graphs
    kernelviz = kernel.getVisualization();
    calculateGP(activeKernels, x_s);
  };
}

// --------------------
// Button Events
// --------------------

function makeActive(kernel, buttonId) {
  buttons = document.getElementsByClassName("kernel-button");
  var i;
  var n_activated = 0;
  var button = document.getElementById(buttonId);

  // Check how many active buttons there are
  for (i = 0; i < buttons.length; i++) {
    n_activated += buttons[i].classList.contains("activated");
  }

  // Deactivate if active (and more than one currently active)
  if (button.classList.contains("activated") && n_activated > 1) {
    button.classList.remove("activated");
    var index = activeKernels.kernels.indexOf(kernel);
    activeKernels.kernels.splice(index, 1);
  }

  // Activate if not active
  else if (!button.classList.contains("activated")) {
    button.classList.add("activated");
    activeKernels.kernels.push(kernel);
  }

  calculateGP(activeKernels, x_s);
}

// --------------------
// Main
// --------------------

var observations = [[], []];

const canvas = document.querySelector("canvas");
var ctx = document.getElementById("myChart").getContext("2d");
var gpChart = makeGPChart(ctx);

// Default slider values, for time being the ranges are set in the html

// Variance, Length
var rbfDefaults = [0.8, 1];

// rbf object, set initial values from defaults
rbf = new RBF(rbfDefaults[0], rbfDefaults[1]);


// Set Up slider listeners
// ---------------------

// Test object to see if we are on kernel settings page
var kernel_settings = document.getElementById("rbfSigmaSlider");

if (typeof kernel_settings != "undefined" && kernel_settings != null) {
  // RBF Options
  var rbfSigmaSlider = document.getElementById("sigma-slider");
  rbfSigmaSlider.value = rbfDefaults[0];

  var rbfLengthSlider = document.getElementById("length-scale-slider");
  rbfLengthSlider.value = rbfDefaults[1];


}
// Initialise GP and HeatMap
x_s = makexPoints(50);
rbfkernelviz = rbf.getVisualization();
activeKernels = new ActiveKernels([rbf], (method = "add"));

if (typeof kernel_settings != "undefined" && kernel_settings != null) {
}
calculateGP(activeKernels, x_s);

// Listen for mouse clicks and update graphs
canvas.addEventListener("mousedown", function (e) {
  addDataPointAtCursor(canvas, gpChart, e);
  calculateGP(activeKernels, x_s);
});

let plotInitialized = false;

function updatePlot() {
    // Example of how you might compute the squared exponential kernel data
    // This is a placeholder function - you'll need to replace it with actual computations
    function computeKernelData(sigma, lengthScale, xRange, yRange) {
        let z = [];
        let x = [];
        let y = [];

        // Generate x and y values based on xRange and yRange
        for (let i = xRange[0]; i <= xRange[1]; i += 0.1) {
            x.push(i);
        }
        for (let i = yRange[0]; i <= yRange[1]; i += 0.1) {
            y.push(i);
        }

        // Compute z values based on the squared exponential kernel
        for (let i = 0; i < y.length; i++) {
            let zRow = [];
            for (let j = 0; j < x.length; j++) {
                let value = sigma * sigma * Math.exp(-Math.pow(x[j] - y[i], 2) / (2 * lengthScale * lengthScale));
                zRow.push(value);
            }
            z.push(zRow);
        }

        return {x, y, z, type: 'surface'};
    }

    var Data = computeKernelData(document.getElementById('sigma-slider').value, document.getElementById('length-scale-slider').value, [0, 10], [0, 10]);
    activeKernels.kernels[0].updateSigma(document.getElementById('sigma-slider').value);
    activeKernels.kernels[0].updateL(document.getElementById('length-scale-slider').value);
    calculateGP(activeKernels, x_s);
    if (!plotInitialized) {
        // Initial plot setup
        let initialData = {
            z: Data.z,
            type: 'surface',
            contours: {
                z: {
                    show:true,
                    usecolormap: true,
                    highlightcolor:"#42f462",
                    project:{z: true}
                }
            }
        };
        let layout = {
            autosize: true,
            scene: {
                xaxis: {title: 'x'}, // Use double backslashes to escape
                yaxis: {title: 'x\''}, // LaTeX for x prime
                zaxis: {title: 'k(x, x′)'} // LaTeX for the kernel function
            },
            font: {
                color: 'black',
                size: 14,
                family: 'Times New Roman'
            }
        };
        Plotly.newPlot('plot', [initialData], layout, {displayModeBar: false});
        plotInitialized = true;
    } else {
        // Update existing plot
        Plotly.restyle('plot', 'z', [Data.z]);
    }
}

// Initial plot
updatePlot();

// Event listeners for sliders
document.getElementById('sigma-slider').addEventListener('input', updatePlot);
document.getElementById('length-scale-slider').addEventListener('input', updatePlot);
document.getElementById('sigma-slider').addEventListener('input', function() {
// Retrieve the current value of the sigma slider
let sigmaValue = this.value;
// Update the label directly with the current value
document.querySelector('label[for="sigma-slider"]').textContent = `Sigma (σ): ${sigmaValue}`;
});

document.getElementById('length-scale-slider').addEventListener('input', function() {
// Retrieve the current value of the length scale slider
let lengthScaleValue = this.value;
// Update the label directly with the current value
document.querySelector('label[for="length-scale-slider"]').textContent = `Length Scale (l): ${lengthScaleValue}`;
});