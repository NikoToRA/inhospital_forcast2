// 数値入力のコントロールボタンの機能
document.addEventListener('DOMContentLoaded', function() {
    // 数値入力フィールドの増減ボタンの設定
    document.querySelectorAll('.input-wrapper').forEach(container => {
        const input = container.querySelector('input[type="number"]');
        const incrementBtn = container.querySelector('.increment');
        const decrementBtn = container.querySelector('.decrement');

        if (input && incrementBtn && decrementBtn) {
            decrementBtn.addEventListener('click', () => {
                const currentValue = parseInt(input.value) || 0;
                input.value = Math.max(0, currentValue - 1);
                saveInputValues();
            });

            incrementBtn.addEventListener('click', () => {
                const currentValue = parseInt(input.value) || 0;
                input.value = currentValue + 1;
                saveInputValues();
            });
        }
    });

    // 日付表示の設定
    const dateInput = document.getElementById('date');
    const dateDisplay = document.querySelector('.date-display');
    
    if (dateInput && dateDisplay) {
        // 初期値を今日の日付に設定
        const today = new Date();
        dateInput.value = today.toISOString().split('T')[0];
        updateDateDisplay(today);

        dateInput.addEventListener('change', (e) => {
            const selectedDate = new Date(e.target.value);
            updateDateDisplay(selectedDate);
        });
    }

    // 前回の入力値を復元
    restoreInputValues();

    // フォームの送信処理
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!validateForm(form)) {
                alert('すべての項目を正しく入力してください。');
                return;
            }

            const formData = new FormData(form);
            
            try {
                const response = await fetch("/predict", {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    // 予測結果を表示
                    const resultContainer = document.getElementById('result');
                    const predictionValue = document.getElementById('prediction-value');
                    const weeklyChart = document.getElementById('weekly-chart');
                    
                    if (predictionValue) {
                        predictionValue.textContent = data.prediction.toFixed(1);
                    }
                    
                    if (weeklyChart) {
                        try {
                            const ctx = weeklyChart.getContext('2d');
                            // 既存のグラフを破棄
                            if (window.weeklyChart) {
                                window.weeklyChart.destroy();
                            }
                            // 新しいグラフを作成
                            window.weeklyChart = new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: data.weekly_predictions.map(p => p.date),
                                    datasets: [{
                                        label: '予測入院患者数',
                                        data: data.weekly_predictions.map(p => p.value),
                                        borderColor: '#0A4B73',
                                        backgroundColor: 'rgba(10, 75, 115, 0.1)',
                                        tension: 0.4,
                                        fill: true
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            display: false
                                        }
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                            title: {
                                                display: true,
                                                text: '入院患者数'
                                            }
                                        }
                                    }
                                }
                            });
                        } catch (chartError) {
                            console.error('Chart error:', chartError);
                            // グラフのエラーは致命的ではないので、エラーメッセージを表示しない
                        }
                    }
                    
                    // 結果セクションを表示
                    if (resultContainer) {
                        resultContainer.style.display = 'block';
                        resultContainer.scrollIntoView({ behavior: 'smooth' });
                    }
                } else {
                    alert('予測に失敗しました: ' + data.message);
                }
            } catch (error) {
                console.error('Error:', error);
                // サーバーエラーやネットワークエラーの場合のみエラーメッセージを表示
                alert('サーバーとの通信中にエラーが発生しました。');
            }
        });
    }
});

// 日付表示の更新
function updateDateDisplay(date) {
    const dateDisplay = document.querySelector('.date-display');
    if (dateDisplay) {
        const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        dateDisplay.textContent = date.toLocaleDateString('ja-JP', options);
    }
}

// フォームのバリデーション
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            field.classList.add('error');
        } else {
            field.classList.remove('error');
        }
    });

    return isValid;
}

// 入力値の保存
function saveInputValues() {
    const form = document.getElementById('prediction-form');
    if (form) {
        const formData = new FormData(form);
        for (let [key, value] of formData.entries()) {
            localStorage.setItem(key, value);
        }
    }
}

// 入力値の復元
function restoreInputValues() {
    const form = document.getElementById('prediction-form');
    if (form) {
        const inputs = form.querySelectorAll('input');
        inputs.forEach(input => {
            const savedValue = localStorage.getItem(input.name);
            if (savedValue) {
                input.value = savedValue;
            }
        });
    }
} 