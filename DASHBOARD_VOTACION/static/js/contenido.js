let votosData = [];
let votosPorRegion = {};
let votosGenerales = {};
let estadisticasPorRegion = {};
let estadisticas = {};
let barChartInstance = null;


function cargarDatos(callback) {
  $.getJSON('/api/estadisticas', function(data) {
    estadisticas = data;
    if (callback) callback();
  });
}

function mostrarGrafico(region = null) {
  let data, titulo, ganador = '';
  let totalVotos = estadisticas.total_votos;
  if (region) {
    // Normaliza el nombre: primera letra mayúscula, resto minúsculas
    region = region.charAt(0).toUpperCase() + region.slice(1).toLowerCase();
  }
  let votosRegion = estadisticas.votos_por_region || {};
  let votosCandidato = estadisticas.votos_por_candidato || {};

  if (region && votosRegion[region]) {
    let regionData = estadisticas.votos_por_candidato_por_region?.[region] || {};
    let candidatos = Object.keys(regionData);
    let votos = Object.values(regionData);
    let maxVotos = Math.max(...votos, 0);
    let idxGanador = votos.indexOf(maxVotos);
    ganador = candidatos[idxGanador] || '';
    titulo = `Votos por candidato en ${region}`;
    data = {
      labels: candidatos,
      datasets: [{
        label: 'Votos',
        data: votos,
        backgroundColor: '#497ca8'
      }]
    };
    // Actualizar total de votos para la región seleccionada
    totalVotos = votosRegion[region];
  } else {
    let candidatos = Object.keys(votosCandidato);
    let votos = Object.values(votosCandidato);
    let maxVotos = Math.max(...votos, 0);
    let idxGanador = votos.indexOf(maxVotos);
    ganador = candidatos[idxGanador] || '';
    titulo = "Votos generales por candidato";
    data = {
      labels: candidatos,
      datasets: [{
        label: 'Votos',
        data: votos,
        backgroundColor: '#497ca8'
      }]
    };
  }

  // Destruir gráfico anterior si existe
  if (barChartInstance) barChartInstance.destroy();

  // Crear nuevo gráfico
  const ctx = document.getElementById('barChart').getContext('2d');
  barChartInstance = new Chart(ctx, {
    type: 'bar',
    data: data,
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: titulo,
          font: { size: 20 }
        },
        legend: { display: false }
      },
      scales: {
        x: { title: { display: true, text: 'Candidato' } },
        y: { title: { display: true, text: 'Votos' }, beginAtZero: true }
      }
    }
  });

  // Mostrar estadísticas y ganador
// Mostrar estadísticas y ganador
// Renderiza los datos principales
document.getElementById('stats-main').innerHTML = `
  <div>
    <span class="stat-icon">🗳️</span>
    <span class="stat-title">Total votos:</span>
    <span class="stat-value">${totalVotos}</span>
  </div>
  <div>
    <span class="stat-icon">⚠️</span>
    <span class="stat-label">Anom. reales:</span>
    <span class="stat-value">${estadisticas.anomalias_reales}</span>
  </div>
  <div>
    <span class="stat-icon">🔍</span>
    <span class="stat-label">Anom. detectadas:</span>
    <span class="stat-value">${estadisticas.anomalias_detectadas}</span>
  </div>
  <div>
    <span class="stat-icon">❌</span>
    <span class="stat-label">Falsos positivos:</span>
    <span class="stat-value">${estadisticas.falsos_positivos}</span>
  </div>
  <div>
    <span class="stat-icon">❓</span>
    <span class="stat-label">Falsos negativos:</span>
    <span class="stat-value">${estadisticas.falsos_negativos}</span>
  </div>
`;

// Renderiza el ganador a la derecha
document.getElementById('stats-ganador').innerHTML = `
  <span class="stat-icon">🏆</span>
  <span class="ganador-label">Candidato ganador:</span>
  <span class="ganador-value">${ganador ? `<span style="color:#2a7d2e">${ganador}</span>` : 'N/A'}</span>
`;
}

function inicializarMapa() {
  var map = L.map('map').setView([-9.19, -75.015], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

  $.getJSON('../static/data/departamentos_perú.geojson', function(geojson) { 
    L.geoJSON(geojson, {
      style: {
        color: "#3388ff",
        weight: 1,
        fillOpacity: 0.1
      },
      onEachFeature: function(feature, layer) {
        // Centroide aproximado
        var latlng;
        if (feature.geometry.type === "Polygon") {
          var coords = feature.geometry.coordinates[0];
          var lat = 0, lng = 0;
          for (var i = 0; i < coords.length; i++) {
            lng += coords[i][0];
            lat += coords[i][1];
          }
          lng /= coords.length;
          lat /= coords.length;
          latlng = [lat, lng];
        } else if (feature.geometry.type === "MultiPolygon") {
          var coords = feature.geometry.coordinates[0][0];
          var lat = 0, lng = 0;
          for (var i = 0; i < coords.length; i++) {
            lng += coords[i][0];
            lat += coords[i][1];
          }
          lng /= coords.length;
          lat /= coords.length;
          latlng = [lat, lng];
        }

        // Marcador en el centroide
        var marker = L.marker(latlng)
          .addTo(map)
          .bindPopup(feature.properties.NOMBDEP);

        marker.on("click", function() {
          document.getElementById("selected-department").innerText = "Departamento seleccionado: " + feature.properties.NOMBDEP;
          mostrarGrafico(feature.properties.NOMBDEP);
        });

        marker.on("popupclose", function() {
          document.getElementById("selected-department").innerText = "Ningún departamento seleccionado";
          mostrarGrafico();
        });
      }
    }).addTo(map);
  });
}

function llenarSelectorCandidatos() {
  const selector = document.getElementById('candidato-selector');
  selector.innerHTML = '<option value="">-- Elegir candidato --</option>';
  let candidatos = Object.keys(estadisticas.votos_por_candidato || {});
  candidatos.forEach(c => {
    let opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    selector.appendChild(opt);
  });
}

function mostrarRegionMasVotos(candidato) {
  if (!candidato) {
    document.getElementById('region-mas-votos').innerText = '';
    return;
  }
  let votosPorRegion = estadisticas.votos_por_candidato_por_region || {};
  let maxVotos = -1;
  let regionGanadora = '';
  Object.entries(votosPorRegion).forEach(([region, votosCands]) => {
    if (votosCands[candidato] !== undefined && votosCands[candidato] > maxVotos) {
      maxVotos = votosCands[candidato];
      regionGanadora = region;
    }
  });
  if (regionGanadora) {
    document.getElementById('region-mas-votos').innerHTML =
      `<b>Región:</b> <span style="color:#2a7d2e">${regionGanadora} (${maxVotos} votos)</span>`;
  } else {
    document.getElementById('region-mas-votos').innerText = 'No hay votos para este candidato.';
  }
}

// Llenar selector y manejar cambios
$(document).ready(function() {
  cargarDatos(function() {
    llenarSelectorCandidatos();
    inicializarMapa();
    mostrarGrafico(); // General al inicio
  });

  $('#candidato-selector').on('change', function() {
    mostrarRegionMasVotos(this.value);
  });
});

// Inicialización
$(document).ready(function() {
  cargarDatos(function() {
    inicializarMapa();
    mostrarGrafico(); // General al inicio
  });
});

// Mostrar datos generales al hacer clic en el botón
$('#btn-general').on('click', function() {
  document.getElementById("selected-department").innerText = "Ningún departamento seleccionado";
  mostrarGrafico();
  $(this).hide();
});
