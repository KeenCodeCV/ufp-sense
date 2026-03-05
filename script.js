// ==========================================
// 1. ระบบจัดการ Loader (หนูแฮมสเตอร์) - ทำงานทันที
// ==========================================
setTimeout(() => {
    const loader = document.getElementById('loader-wrapper');
    if (loader) {
        loader.style.opacity = '0'; 
        setTimeout(() => {
            loader.style.display = 'none'; 
            loader.remove(); 
        }, 500);
    }
}, 1500); 

// ==========================================
// 2. ระบบเวลา (Time Logic)
// ==========================================
function updateTime() {
    const now = new Date();
    const dateElem = document.getElementById('currentDate');
    const timeElem = document.getElementById('currentTime');
    if (dateElem) {
        dateElem.textContent = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    }
    if (timeElem) {
        timeElem.textContent = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
    }
}
setInterval(updateTime, 1000);
setTimeout(updateTime, 100); 

// ==========================================
// 3. ระบบจัดการสถานะ (Status & AI Logic)
// ==========================================
window.updateStatus = function (value, aiText = null) {
    const middleWrapper = document.getElementById('middleWrapper');
    const iconBoxBg = document.getElementById('iconBoxBg');
    const indoorIcon = document.getElementById('indoorIcon');
    const pm01Value = document.getElementById('pm01Value');
    const pm01Unit = document.getElementById('pm01Unit');
    const pm01StatusPill = document.getElementById('pm01StatusPill');
    const pm01StatusIcon = document.getElementById('pm01StatusIcon');
    const pm01StatusText = document.getElementById('pm01StatusText');
    const alertBoxBg = document.getElementById('alertBoxBg');
    const suggestionText = document.getElementById('suggestionText');
    const alertIcon = document.getElementById('alertIcon');

    if (pm01Value) pm01Value.textContent = value.toLocaleString();

    // 👉 ดักจับกรณีไม่มีข้อความ AI ส่งมา (เช่น ตอนกดปุ่ม Test System)
    if (!aiText) {
        if (value >= 20000) aiText = "🚨 สูงมาก (แดง): ค่าฝุ่นสะสมระดับวิกฤต (PM0.1 > 20,000) ควรงดเข้าพื้นที่ หรือสวมหน้ากากกรองอนุภาคขั้นสูงทันที";
        else if (value >= 10000) aiText = "🟠 สูง (ส้ม): ค่าฝุ่นระดับสีส้มเริ่มส่งผลกระทบต่อสุขภาพ ควรเปิดเครื่องฟอกอากาศและหลีกเลี่ยงการอยู่ในพื้นที่นานเกินไป";
        else if (value >= 1000) aiText = "🟡 ปานกลาง (เหลือง): เริ่มมีแนวโน้มกักเก็บฝุ่น ควรระบายอากาศหากทำได้";
        else aiText = "🟢 ต่ำ (เขียว): ปลอดภัย ระดับอนุภาคฝุ่นอยู่ในเกณฑ์ดีเยี่ยม อากาศถ่ายเทได้ดี";
    }

    const allBgs = ['bg-[#5dbb47]', 'bg-[#f0b100]', 'bg-[#f97316]', 'bg-[#d93a3a]', 'bg-slate-300'];
    const allLightBgs = ['bg-[#eef8eb]', 'bg-[#fef0b8]', 'bg-[#ffedd5]', 'bg-[#f8d7d7]', 'bg-slate-50'];
    const allTexts = ['text-[#5dbb47]', 'text-[#f0b100]', 'text-[#f97316]', 'text-[#d93a3a]', 'text-slate-400', 'text-slate-500'];

    if (middleWrapper) middleWrapper.classList.remove(...allLightBgs);
    if (alertBoxBg) alertBoxBg.classList.remove(...allLightBgs);
    if (iconBoxBg) iconBoxBg.classList.remove(...allBgs);
    if (pm01Value) pm01Value.classList.remove(...allTexts);
    if (pm01Unit) pm01Unit.classList.remove(...allTexts);
    if (pm01StatusPill) pm01StatusPill.classList.remove(...allTexts, 'animate-pulse');
    if (pm01StatusText) pm01StatusText.classList.remove('text-blink');

    if (value < 1000) {
        if (middleWrapper) middleWrapper.classList.add('bg-[#eef8eb]');
        if (alertBoxBg) alertBoxBg.classList.add('bg-[#eef8eb]');
        if (iconBoxBg) iconBoxBg.classList.add('bg-[#5dbb47]');
        if (pm01Value) pm01Value.classList.add('text-[#5dbb47]');
        if (pm01Unit) pm01Unit.classList.add('text-[#5dbb47]');
        if (pm01StatusPill) pm01StatusPill.classList.add('text-[#5dbb47]');

        if (indoorIcon) indoorIcon.src = 'https://img5.pic.in.th/file/secure-sv1/Safe123.png';
        if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-circle-check mr-2 text-lg';
        if (pm01StatusText) pm01StatusText.innerText = 'Status: Safe';
        if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-[#5dbb47] text-xl animate-bounce';
        
    } else if (value >= 1000 && value < 10000) {
        if (middleWrapper) middleWrapper.classList.add('bg-[#fef0b8]');
        if (alertBoxBg) alertBoxBg.classList.add('bg-[#fef0b8]');
        if (iconBoxBg) iconBoxBg.classList.add('bg-[#f0b100]');
        if (pm01Value) pm01Value.classList.add('text-[#f0b100]');
        if (pm01Unit) pm01Unit.classList.add('text-[#f0b100]');
        if (pm01StatusPill) pm01StatusPill.classList.add('text-[#f0b100]');

        if (indoorIcon) indoorIcon.src = 'https://img2.pic.in.th/Warning123.png';
        if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-triangle-exclamation mr-2 text-lg';
        if (pm01StatusText) pm01StatusText.innerText = 'Status: Moderate';
        if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-[#f0b100] text-xl animate-bounce';
        
    } else if (value >= 10000 && value < 20000) {
        if (middleWrapper) middleWrapper.classList.add('bg-[#ffedd5]');
        if (alertBoxBg) alertBoxBg.classList.add('bg-[#ffedd5]');
        if (iconBoxBg) iconBoxBg.classList.add('bg-[#f97316]');
        if (pm01Value) pm01Value.classList.add('text-[#f97316]');
        if (pm01Unit) pm01Unit.classList.add('text-[#f97316]');
        if (pm01StatusPill) pm01StatusPill.classList.add('text-[#f97316]');

        if (indoorIcon) indoorIcon.src = 'https://img2.pic.in.th/641319986_1432330355266272_9107297040447500607_n-removebg-preview.png';
        if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-bell mr-2 text-lg';
        if (pm01StatusText) {
            pm01StatusText.innerText = 'Status: High';
            pm01StatusText.classList.add('text-blink');
        }
        if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-[#f97316] text-xl animate-bounce';
        
    } else {
        if (middleWrapper) middleWrapper.classList.add('bg-[#f8d7d7]');
        if (alertBoxBg) alertBoxBg.classList.add('bg-[#f8d7d7]');
        if (iconBoxBg) iconBoxBg.classList.add('bg-[#d93a3a]');
        if (pm01Value) pm01Value.classList.add('text-[#d93a3a]');
        if (pm01Unit) pm01Unit.classList.add('text-[#d93a3a]');
        if (pm01StatusPill) pm01StatusPill.classList.add('text-[#d93a3a]');

        if (indoorIcon) indoorIcon.src = 'https://img2.pic.in.th/Danger123.png';
        if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-skull-crossbones mr-2 text-lg';
        if (pm01StatusText) {
            pm01StatusText.innerText = 'Status: Danger';
            pm01StatusText.classList.add('text-blink');
        }
        if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-[#d93a3a] text-xl animate-bounce';
    }

    if (suggestionText) suggestionText.textContent = aiText;
}

// ==========================================
// 4. ระบบจัดการเข็มทิศทิศทางลม
// ==========================================
window.updateWindDirection = function (degree) {
    const compass = document.getElementById('windCompass');
    const degVal = document.getElementById('val-wind-dir-deg');
    const nameVal = document.getElementById('val-wind-dir-name');

    if (compass) compass.style.transform = `rotate(${degree}deg)`;

    if (degVal && nameVal) {
        const directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
        const index = Math.floor(((degree % 360) + 11.25) / 22.5) % 16;
        degVal.innerText = `${degree}°`;
        nameVal.innerText = directions[index];
    }
}

// ==========================================
// 5. ระบบกราฟ (Dynamic Chart.js)
// ==========================================
let chartCurrentObj, chartHourObj, chartDayObj;
const commonOptions = {
    responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } },
    scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#f3f4f6' } } },
    elements: { line: { tension: 0.4 }, point: { radius: 3, hoverRadius: 5 } }
};

window.updateCharts = function (arrCurrent, arrHour, arrDay) {
    if (chartCurrentObj) chartCurrentObj.destroy();
    if (chartHourObj) chartHourObj.destroy();
    if (chartDayObj) chartDayObj.destroy();

    const avgHour = arrHour.reduce((a, b) => a + b, 0) / arrHour.length;
    const avgDay = arrDay.reduce((a, b) => a + b, 0) / arrDay.length;

    const colorHour = avgHour > 20000 ? '#d93a3a' : '#10b981';
    const bgHour = avgHour > 20000 ? 'rgba(217, 58, 58, 0.1)' : 'rgba(16, 185, 129, 0.1)';

    const colorDay = avgDay > 10000 ? '#d93a3a' : '#f59e0b';
    const bgDay = avgDay > 10000 ? 'rgba(217, 58, 58, 0.1)' : 'rgba(245, 158, 11, 0.1)';

    const ctx1 = document.getElementById('chartCurrent');
    if (ctx1) chartCurrentObj = new Chart(ctx1, { type: 'line', data: { labels: ['t-25', 't-20', 't-15', 't-10', 't-5', 'Live'], datasets: [{ data: arrCurrent, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, borderWidth: 2 }] }, options: commonOptions });

    const ctx2 = document.getElementById('chartHour');
    if (ctx2) chartHourObj = new Chart(ctx2, { type: 'line', data: { labels: ['H-4', 'H-3', 'H-2', 'H-1', 'Now'], datasets: [{ data: arrHour, borderColor: colorHour, backgroundColor: bgHour, fill: true, borderWidth: 2 }] }, options: commonOptions });

    const ctx3 = document.getElementById('chartDay');
    if (ctx3) chartDayObj = new Chart(ctx3, { type: 'line', data: { labels: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'D-1', 'Today'], datasets: [{ data: arrDay, borderColor: colorDay, backgroundColor: bgDay, fill: true, borderWidth: 2 }] }, options: commonOptions });
}

setTimeout(() => {
    if (window.updateCharts) updateCharts([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]);
}, 200);

// ==========================================
// 6. Test Mode & Reset System
// ==========================================
let testInterval;
window.startTestMode = function () {
    clearInterval(testInterval);
    console.log("Starting Test Mode...");
    simulateStep();
    testInterval = setInterval(simulateStep, 2500);
}

function simulateStep() {
    const rPM01 = Math.floor(Math.random() * 35000); 
    const rPM25 = (Math.random() * 50).toFixed(1);
    const rTemp = (25 + Math.random() * 10).toFixed(1);
    const rHumid = (50 + Math.random() * 30).toFixed(1);
    const rWindDir = Math.floor(Math.random() * 360);

    if (window.updateStatus) window.updateStatus(rPM01);

    if (window.updateCharts) {
        window.updateCharts(
            [rPM01 * 0.8, rPM01 * 1.1, rPM01 * 0.9, rPM01 * 1.05, rPM01 * 0.95, rPM01],
            [rPM01 * 1.2, rPM01 * 1.1, rPM01 * 1.0, rPM01 * 0.9, rPM01],
            [rPM01 * 0.5, rPM01 * 0.7, rPM01 * 1.5, rPM01 * 1.2, rPM01 * 0.8, rPM01 * 1.1, rPM01]
        );
    }

    updateText('val-pm25', rPM25);
    updateText('val-temp', rTemp);
    updateText('val-humid', rHumid);
    if (window.updateWindDirection) window.updateWindDirection(rWindDir);
}

window.resetSystem = function () {
    clearInterval(testInterval);

    const middleWrapper = document.getElementById('middleWrapper');
    const iconBoxBg = document.getElementById('iconBoxBg');
    const indoorIcon = document.getElementById('indoorIcon');
    const pm01Value = document.getElementById('pm01Value');
    const pm01Unit = document.getElementById('pm01Unit');
    const pm01StatusPill = document.getElementById('pm01StatusPill');
    const pm01StatusIcon = document.getElementById('pm01StatusIcon');
    const pm01StatusText = document.getElementById('pm01StatusText');
    const alertBoxBg = document.getElementById('alertBoxBg');
    const suggestionText = document.getElementById('suggestionText');
    const alertIcon = document.getElementById('alertIcon');

    // 👉 ล็อกให้ Real-time คงความสีเขียวไว้
    const realTimeIndicator = document.getElementById('realTimeIndicator');
    const rtPing = document.getElementById('rtPing');
    const rtDot = document.getElementById('rtDot');

    if (realTimeIndicator) {
        realTimeIndicator.className = 'bg-green-50 text-green-600 px-4 py-1.5 rounded-full text-[13px] font-bold flex items-center gap-2 border border-green-100 transition-colors duration-300';
    }
    if (rtPing) rtPing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75';
    if (rtDot) rtDot.className = 'relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500';

    const allBgs = ['bg-[#5dbb47]', 'bg-[#f0b100]', 'bg-[#d93a3a]', 'bg-slate-300'];
    const allLightBgs = ['bg-[#eef8eb]', 'bg-[#fef0b8]', 'bg-[#f8d7d7]', 'bg-slate-50'];
    const allTexts = ['text-[#5dbb47]', 'text-[#f0b100]', 'text-[#d93a3a]', 'text-slate-400', 'text-slate-500', 'text-slate-700'];

    if (middleWrapper) {
        middleWrapper.classList.remove(...allLightBgs);
        middleWrapper.classList.add('bg-slate-50');
    }
    if (alertBoxBg) {
        alertBoxBg.classList.remove(...allLightBgs);
        alertBoxBg.classList.add('bg-slate-50');
    }
    if (iconBoxBg) {
        iconBoxBg.classList.remove(...allBgs);
        iconBoxBg.classList.add('bg-slate-300'); 
    }
    if (pm01Value) {
        pm01Value.innerText = '--';
        pm01Value.classList.remove(...allTexts);
        pm01Value.classList.add('text-slate-400'); 
    }
    if (pm01Unit) {
        pm01Unit.classList.remove(...allTexts);
        pm01Unit.classList.add('text-slate-400'); 
    }
    if (pm01StatusPill) {
        pm01StatusPill.classList.remove(...allTexts);
        pm01StatusPill.classList.add('text-slate-500'); 
    }

    if (indoorIcon) indoorIcon.src = 'https://img5.pic.in.th/file/secure-sv1/Safe123.png'; 
    if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-spinner fa-spin mr-2 text-lg';
    if (pm01StatusText) {
        pm01StatusText.innerText = 'Status: Waiting...';
        pm01StatusText.classList.remove('text-blink'); 
    }
    if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-slate-400 text-xl animate-bounce'; 
    if (suggestionText) suggestionText.innerText = 'กำลังรอรับข้อมูลจากเซ็นเซอร์เพื่อทำการวิเคราะห์สภาพแวดล้อม...';

    updateText('val-pm25', '--');
    updateText('val-temp', '--');
    updateText('val-humid', '--');
    if (window.updateWindDirection) window.updateWindDirection(0);
    document.getElementById('val-wind-dir-deg').innerText = '--°';
    document.getElementById('val-wind-dir-name').innerText = '';

    if (window.updateCharts) updateCharts([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]);
}

function updateText(id, val) {
    const el = document.getElementById(id);
    if (el) el.innerText = val;
}