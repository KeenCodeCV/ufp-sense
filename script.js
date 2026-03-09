// ==========================================
// 1. ระบบจัดการ Loader (หนูแฮมสเตอร์) - ทำงานทันที
// ==========================================
setTimeout(() => {
    const loader = document.getElementById('loader-wrapper');
    if (loader) {
        loader.style.opacity = '0'; // เฟดให้จางลง
        setTimeout(() => {
            loader.style.display = 'none'; // ซ่อนออกจากหน้าจอให้คลิกทะลุได้
            loader.remove(); // ลบออกจาก HTML
        }, 500);
    }
}, 1500); // แสดงหนูแฮมสเตอร์ 1.5 วินาที

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
setTimeout(updateTime, 100); // รันครั้งแรกทันที

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

    // 👉 เมื่อมีข้อมูลเข้ามา ให้เปลี่ยนปุ่ม Real-time เป็นสีเขียว
    const realTimeIndicator = document.getElementById('realTimeIndicator');
    const rtPing = document.getElementById('rtPing');
    const rtDot = document.getElementById('rtDot');
    
    if (realTimeIndicator) {
        realTimeIndicator.classList.remove('bg-red-50', 'text-red-500', 'border-red-100');
        realTimeIndicator.classList.add('bg-green-50', 'text-green-600', 'border-green-100');
    }
    if (rtPing) rtPing.classList.replace('bg-red-400', 'bg-green-400');
    if (rtDot) rtDot.classList.replace('bg-red-500', 'bg-green-500');

    // 👉 ดักจับกรณีไม่มีข้อความ AI ส่งมา (ตอนกด Test System) ให้ใช้คำภาษาอังกฤษ
    if (!aiText) {
        if (value >= 20000) aiText = "Hazardous air conditions detected. Please remain indoors, keep all windows closed, and use high-efficiency air purifiers immediately.";
        else if (value >= 10000) aiText = "Air quality is noticeably reduced. It is highly recommended to turn on air purifiers and minimize outdoor activities.";
        else if (value >= 1000) aiText = "Air quality is acceptable. However, sensitive individuals should monitor for any discomfort and consider limiting prolonged outdoor exertion.";
        else aiText = "The air quality is excellent. The environment is safe, making it a great time for normal activities and natural ventilation.";
    }

    // เพิ่มชุดสีเข้าไป
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

// ===================================
    // เงื่อนไข 4 ระดับ (อัปเดตเกณฑ์ใหม่ล่าสุด!)
    // ===================================
    if (value <= 999) {
        // === Safe (เขียว: <= 999) ===
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

    } else if (value >= 1000 && value <= 9999) {
        // === Moderate (เหลือง: 1,000 - 9,999) ===
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

    } else if (value >= 10000 && value <= 19999) {
        // === High (ส้ม: 10,000 - 19,999) ===
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
        // === Danger (แดง: 20,000 ขึ้นไป) ===
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
    
    // อัปเดตข้อความ AI
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
    // เปลี่ยน t-25... เป็น Min-25... และ Live เป็น Now
    if (ctx1) chartCurrentObj = new Chart(ctx1, { type: 'line', data: { labels: ['Min-25', 'Min-20', 'Min-15', 'Min-10', 'Min-5', 'Now'], datasets: [{ data: arrCurrent, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, borderWidth: 2 }] }, options: commonOptions });

    const ctx2 = document.getElementById('chartHour');
    // กราฟชั่วโมงมี Now อยู่แล้ว ไม่ต้องเปลี่ยน
    if (ctx2) chartHourObj = new Chart(ctx2, { type: 'line', data: { labels: ['H-4', 'H-3', 'H-2', 'H-1', 'Now'], datasets: [{ data: arrHour, borderColor: colorHour, backgroundColor: bgHour, fill: true, borderWidth: 2 }] }, options: commonOptions });

    const ctx3 = document.getElementById('chartDay');
    // เปลี่ยน Today เป็น Now
    if (ctx3) chartDayObj = new Chart(ctx3, { type: 'line', data: { labels: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'D-1', 'Now'], datasets: [{ data: arrDay, borderColor: colorDay, backgroundColor: bgDay, fill: true, borderWidth: 2 }] }, options: commonOptions });}

// โหลดกราฟตั้งต้นตอนเปิดเว็บ
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

    // 👉 คืนค่าปุ่ม Real-time กลับเป็นสีแดง (สถานะรอข้อมูล)
    const realTimeIndicator = document.getElementById('realTimeIndicator');
    const rtPing = document.getElementById('rtPing');
    const rtDot = document.getElementById('rtDot');

    if (realTimeIndicator) {
        realTimeIndicator.classList.remove('bg-green-50', 'text-green-600', 'border-green-100');
        realTimeIndicator.classList.add('bg-red-50', 'text-red-500', 'border-red-100');
    }
    if (rtPing) rtPing.classList.replace('bg-green-400', 'bg-red-400');
    if (rtDot) rtDot.classList.replace('bg-green-500', 'bg-red-500');

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
        iconBoxBg.classList.add('bg-slate-300'); // กล่องหน้าคนเป็นสีเทา
    }
    if (pm01Value) {
        pm01Value.innerText = '--';
        pm01Value.classList.remove(...allTexts);
        pm01Value.classList.add('text-slate-400'); // ตัวเลขเป็นสีเทา
    }
    if (pm01Unit) {
        pm01Unit.classList.remove(...allTexts);
        pm01Unit.classList.add('text-slate-400'); // หน่วยเป็นสีเทา
    }
    if (pm01StatusPill) {
        pm01StatusPill.classList.remove(...allTexts);
        pm01StatusPill.classList.add('text-slate-500'); // ป้ายสถานะเป็นสีเทา
    }

    if (indoorIcon) indoorIcon.src = 'https://img5.pic.in.th/file/secure-sv1/Safe123.png'; // กลับมาหน้ายิ้ม
    if (pm01StatusIcon) pm01StatusIcon.className = 'fa-solid fa-spinner fa-spin mr-2 text-lg';
    if (pm01StatusText) {
        pm01StatusText.innerText = 'Status: Waiting...';
        pm01StatusText.classList.remove('text-blink'); // เอากระพริบออก
    }
    if (alertIcon) alertIcon.className = 'fa-solid fa-robot text-slate-400 text-xl animate-bounce'; // หุ่นยนต์สีเทา
    
    // 👉 เปลี่ยนข้อความรอโหลดเป็นภาษาอังกฤษ
    if (suggestionText) suggestionText.innerText = 'Waiting for sensor data to analyze the environmental conditions...';

    // รีเซ็ตค่าการ์ดด้านนอก
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