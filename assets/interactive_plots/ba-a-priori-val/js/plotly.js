async function fetchData(filename) {
    try {
        const response = await fetch(filename);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}
var totalSamples = 0; 
var tracesAdded = 0;


async function initializePlot() {
    // Initialize data
    const urlParams = new URLSearchParams(window.location.search);
    const L_f = urlParams.get('L_f');
    const f_bar = urlParams.get('f_bar');
    // Select the div Element
    const divHeader = document.querySelector('.header');

    // Modify the div Content
    if (divHeader && L_f !== null && f_bar !== null) {
        divHeader.innerHTML = `Gaussian Process with \\(L_{max} = ${L_f}\\) and \\(M_{supp} = ${f_bar}\\)`;
    } else {
        console.log("parameter is null")
    }
    var X_new = await fetchData(`data/L_f${L_f}f_bar${f_bar}/X_new.json`);
    var y_mean = await fetchData(`data/L_f${L_f}f_bar${f_bar}/y_mean.json`);
    var y_lower = await fetchData(`data/L_f${L_f}f_bar${f_bar}/y_lower.json`);
    var y_upper = await fetchData(`data/L_f${L_f}f_bar${f_bar}/y_upper.json`);
    var y_samples = await fetchData(`data/L_f${L_f}f_bar${f_bar}/y_samples.json`);
    var slope_violation_proba = await fetchData(`data/L_f${L_f}f_bar${f_bar}/slope_violation_proba.json`);
    var max_lipschitz = await fetchData(`data/L_f${L_f}f_bar${f_bar}/max_lipschitz.json`);
    var max_deviation = await fetchData(`data/L_f${L_f}f_bar${f_bar}/max_deviation.json`);
    var bound_violation_proba = await fetchData(`data/L_f${L_f}f_bar${f_bar}/bound_violation_proba.json`);




    // Create initial plot
    var trace1 = {
        x: X_new,
        y: y_mean,
        mode: 'lines',
        name: 'GP mean',
        line: {color: "#708090"}
    };

    var trace2 = {
        x: X_new,
        y: y_lower,
        mode: 'lines',
        name: 'Lower bound',
        fill: 'tozeroy',
        fillcolor: 'rgba(135, 206, 235, 0.4)',
        line: {
            color: 'rgba(135, 206, 235, 0.7)'
        }

    };

    var trace3 = {
        x: X_new,
        y: y_upper,
        mode: 'lines',
        name: 'Upper bound',
        fill: 'tozeroy',
        fillcolor: 'rgba(135, 206, 235, 0.4)',
        line: {
            color: 'rgba(135, 206, 235, 0.7)'
        }
    };

    var layout = {
        showlegend: false,
        dragmode: false,
        hovermode: false,
        autosize: true,
        xaxis: {
            title: 'x',
            fixedrange: true
        },
        yaxis: {
            title: 'f(.)',
            fixedrange: true
        },
        shapes: [
            // Line at y = 1
            {
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper', // 'paper' refers to the entire range of the x-axis
                y0: f_bar,
                y1: f_bar,
                yref: 'y',
                line: {
                    color: 'red',
                    width: 2,
                    dash: 'dot' // Optional: makes the line dotted
                }
            },
            // Line at y = -1
            {
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper', // 'paper' refers to the entire range of the x-axis
                y0: -f_bar,
                y1: -f_bar,
                yref: 'y',
                line: {
                    color: 'red',
                    width: 2,
                    dash: 'dot' // Optional: makes the line dotted
                }
            }
        ]
    };

    Plotly.newPlot('myDiv', [trace1, trace2, trace3], layout, {displayModeBar: false, responsive: true});
    // Function to add samples
    function addSamples() {
        totalSamples = totalSamples === 0 ? 8 : Math.min(totalSamples * 2, y_samples.length);
        let newTraces = []; // Initialize an array to hold all new traces

        for (let i = tracesAdded; i < totalSamples; i++) {
            // Assuming y_samples is an array of arrays with your sample functions
            var sampleTrace = {
                x: X_new,
                y: y_samples[i],
                mode: 'lines',
                line: {width: 1},
                hoverinfo: 'none', // Optional: disable hover info
                showlegend: false // Ensure the trace does not appear in the legend
            };
        
            newTraces.push(sampleTrace); // Add the new trace to the array
        }

        Plotly.addTraces('myDiv', newTraces);

        if (totalSamples >= y_samples.length) {
            // Disable the button or handle the case where all samples are added
        }
        tracesAdded = totalSamples;
        // Update the title to reflect the number of samples
        Plotly.relayout('myDiv');
        // Update content using JavaScript
        document.getElementById("sample-count").textContent = "Functions Sampled: " + totalSamples;
        document.getElementById("supremum-violation").textContent = "Supremum violation: " + bound_violation_proba[totalSamples-1].toFixed(2);
        document.getElementById("lipschitz-violation").textContent = "Lipschitz constant violation: " + slope_violation_proba[totalSamples-1].toFixed(2);
        document.getElementById("max-absolute").textContent = "Maximum absolute value: " + max_deviation[totalSamples-1].toFixed(2);
        document.getElementById("max-lipschitz").textContent = "Maximum Lipschitz constant: " + max_lipschitz[totalSamples-1].toFixed(2);
    }

    // Function to reset the plot
    function resetPlot() {
        Plotly.purge('myDiv');
        Plotly.newPlot('myDiv', [trace1, trace2, trace3], layout);
        totalSamples = 0; // Reset the total number of samples added
        tracesAdded = 0;
        addSamples();
    }
    // Set up event listeners for buttons
    addSamples();
    document.getElementById('addSamplesBtn').addEventListener('click', addSamples, { passive: true });
    document.getElementById('resetBtn').addEventListener('click', resetPlot, { passive: true });
};

initializePlot();

