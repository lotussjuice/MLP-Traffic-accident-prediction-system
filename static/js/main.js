function toggleHourInput() {
    var checkbox = document.getElementById('checkDia');
    var group = document.getElementById('hourInputGroup');
    if (checkbox.checked) {
        group.style.opacity = '0.3';
        group.style.pointerEvents = 'none';
    } else {
        group.style.opacity = '1';
        group.style.pointerEvents = 'auto';
    }
}
toggleHourInput();

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function toggleMapLayer() {
    var iframe = document.querySelector('iframe');
    if (iframe && iframe.contentWindow && typeof iframe.contentWindow.toggleLayerGlobal === "function") {
        iframe.contentWindow.toggleLayerGlobal();
        var btnText = document.getElementById("btnText");
        if (btnText.innerText.includes("Ocultar")) {
            btnText.innerText = "Mostrar Capa";
        } else {
            btnText.innerText = "Ocultar Capa";
        }
    }
}

function centerMap() {
    var iframe = document.querySelector('iframe');
    if (iframe && iframe.contentWindow && typeof iframe.contentWindow.centerMapGlobal === "function") {
        iframe.contentWindow.centerMapGlobal();
    }
}

// Funciones para Sidebar Derecha
function openSidebar() {
    document.getElementById("rightSidebar").classList.add("active");
}

function closeSidebar() {
    document.getElementById("rightSidebar").classList.remove("active");
}

function loadAccidents(lat, lng) {
    openSidebar();
    var contentDiv = document.getElementById("accidentsContent");
    contentDiv.innerHTML = '<div class="text-center mt-5"><div class="spinner-border text-light"></div><p class="mt-2">Cargando datos...</p></div>';
    
    // Fetch a la nueva ruta de flask
    fetch('/get_accidents', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({lat: lat, lng: lng})
    })
    .then(response => response.json())
    .then(data => {
        if(data.error) {
            contentDiv.innerHTML = '<p class="text-danger">Error: ' + data.error + '</p>';
            return;
        }
        if(data.length === 0) {
             contentDiv.innerHTML = '<p class="text-muted">No hay registros históricos exactos en este cuadrante, pero el modelo predice riesgo por proximidad.</p>';
             return;
        }
        
        var html = '<p class="small text-info mb-3">' + data.length + ' accidentes encontrados.</p>';
        data.forEach(acc => {
            // Fecha, Hora y Clima traducido
            html += `
            <div class="acc-item">
                <div class="d-flex justify-content-between mb-2">
                    <span class="badge bg-secondary"><i class="bi bi-calendar-event"></i> ${acc.date}</span>
                    <span class="badge bg-dark border border-secondary"><i class="bi bi-clock"></i> ${acc.time}</span>
                </div>
                <div class="acc-desc">${acc.desc}</div>
                <div class="small text-info mt-2"><i class="bi bi-cloud"></i> ${acc.weather}</div>
            </div>
            `;
        });
        contentDiv.innerHTML = html;
    })
    .catch(err => {
        console.error(err);
        contentDiv.innerHTML = '<p class="text-danger">Error de conexión.</p>';
    });
}

// Funciones Dashboard
function openDashboard() {
    document.getElementById('dashboardOverlay').style.display = 'flex';
    
    fetch('/get_dashboard_stats')
    .then(r => r.json())
    .then(data => {
        if(data.error) {
            document.getElementById('dashboardBody').innerHTML = '<h3 class="text-danger text-center">Error cargando datos</h3>';
            return;
        }
        
        // Renderizar Stats
        let colorGrowth = data.growth >= 0 ? 'growth-pos' : 'growth-neg';
        let sign = data.growth >= 0 ? '+' : '';
        
        // Construccion de items top weather
        let weatherList = '';
        data.top_weather.forEach(w => {
             weatherList += `<li class="list-group-item bg-transparent text-white border-secondary d-flex justify-content-between">
                <span>${w[0]}</span> <span class="badge bg-secondary">${w[1]}</span>
             </li>`;
        });

        let html = `
            <div class="row g-4 mb-4">
                <div class="col-md-4">
                    <div class="stat-card">
                        <div class="stat-val">${data.total.toLocaleString()}</div>
                        <div class="stat-label">Total Accidentes</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stat-card">
                        <div class="stat-val ${colorGrowth}">${sign}${data.growth.toFixed(1)}%</div>
                        <div class="stat-label">Tasa Crecimiento (Año/Año)</div>
                    </div>
                </div>
                <div class="col-md-4">
                     <div class="stat-card">
                        <div class="stat-val text-warning">${data.most_common_day}</div>
                        <div class="stat-label">Día más peligroso</div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <h5 class="mb-3 text-info">Top Climas Accidentados</h5>
                    <ul class="list-group list-group-flush border-start border-secondary ps-3">
                        ${weatherList}
                    </ul>
                </div>
                 <div class="col-md-6">
                    <h5 class="mb-3 text-info">Distribución Horaria</h5>
                    <div class="progress mb-2" style="height: 25px;">
                        <div class="progress-bar bg-warning" role="progressbar" style="width: ${data.day_pct}%">Día (${data.day_pct}%)</div>
                        <div class="progress-bar bg-primary" role="progressbar" style="width: ${100-data.day_pct}%">Noche (${(100-data.day_pct).toFixed(0)}%)</div>
                    </div>
                    <small class="text-muted">Accidentes ocurridos entre 06:00 y 18:00 vs resto.</small>
                </div>
            </div>
        `;
        document.getElementById('dashboardBody').innerHTML = html;
    });
}

function closeDashboard() {
    document.getElementById('dashboardOverlay').style.display = 'none';
}