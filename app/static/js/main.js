// 数値入力のコントロールボタンの機能
document.addEventListener('DOMContentLoaded', function() {
    // 数値入力フィールドの増減ボタンの設定
    document.querySelectorAll('.input-wrapper').forEach(container => {
        const input = container.querySelector('input[type="number"]');
        const incrementBtn = container.querySelector('.increment-btn');
        const decrementBtn = container.querySelector('.decrement-btn');

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
    const dateInput = document.getElementById('prediction-date');
    const weekdayDisplay = document.getElementById('weekday-display');
    
    if (dateInput && weekdayDisplay) {
        // 初期値を今日の日付に設定
        const today = new Date();
        dateInput.value = today.toISOString().split('T')[0];
        updateWeekdayDisplay();

        dateInput.addEventListener('change', updateWeekdayDisplay);
    }

    // 前回の入力値を復元
    restoreInputValues();

    // 予測ボタンのクリックイベント
    document.getElementById('predict-button').addEventListener('click', async function() {
        try {
            // 入力値の検証
            const dateInput = document.getElementById('prediction-date');
            const totalOutpatientInput = document.getElementById('total-outpatient');
            const introOutpatientInput = document.getElementById('intro-outpatient');
            const erPatientsInput = document.getElementById('er-patients');
            const bedCountInput = document.getElementById('bed_count');
            const previousInpatientInput = document.getElementById('previous-inpatient');

            // 必須フィールドのチェック
            if (!dateInput.value) {
                alert('予測日を入力してください。');
                return;
            }

            // 数値の範囲チェック
            const formData = {
                date: dateInput.value,
                total_outpatient: parseInt(totalOutpatientInput.value) || 0,
                intro_outpatient: parseInt(introOutpatientInput.value) || 0,
                er_patients: parseInt(erPatientsInput.value) || 0,
                bed_count: parseInt(bedCountInput.value) || 0,
                y: parseInt(previousInpatientInput.value) || 0  // 前日の新規入院患者数
            };

            // 値の範囲チェック
            if (formData.total_outpatient < 0 || formData.total_outpatient > 2000) {
                alert('外来患者数は0から2000の間で入力してください。');
                return;
            }
            if (formData.intro_outpatient < 0 || formData.intro_outpatient > 200) {
                alert('紹介患者数は0から200の間で入力してください。');
                return;
            }
            if (formData.er_patients < 0 || formData.er_patients > 100) {
                alert('救急搬送患者数は0から100の間で入力してください。');
                return;
            }
            if (formData.bed_count < 0 || formData.bed_count > 1000) {
                alert('入院患者数は0から1000の間で入力してください。');
                return;
            }
            if (formData.y < 0 || formData.y > 100) {
                alert('前日の新規入院患者数は0から100の間で入力してください。');
                return;
            }

            console.log('送信するデータ:', formData);  // デバッグ出力

            // Content-Typeを明示的に指定
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            console.log('予測レスポンス:', result);

            if (result.status === 'error') {
                console.error('予測エラー:', result);
                let errorMessage = result.message || '予測の実行中にエラーが発生しました。';
                alert(errorMessage);
                return;
            }

            // 予測結果の表示
            const resultDiv = document.getElementById('prediction-result');
            const dailyPrediction = document.getElementById('daily-prediction');
            
            if (result.prediction !== undefined && dailyPrediction) {
                dailyPrediction.textContent = `${result.prediction.toFixed(1)}人`;
                resultDiv.style.display = 'block';

                // 週間予測の取得と表示
                try {
                    const weeklyResponse = await fetch('/predict/weekly', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        },
                        body: JSON.stringify({
                            date: formData.date,
                            bed_count: formData.bed_count
                        })
                    });

                    const weeklyResult = await weeklyResponse.json();
                    console.log('週間予測レスポンス:', weeklyResult);

                    if (weeklyResult.status === 'success' && weeklyResult.predictions) {
                        drawWeeklyPredictionChart(weeklyResult.predictions);
                    } else {
                        throw new Error(weeklyResult.message || '週間予測の取得に失敗しました。');
                    }
                } catch (error) {
                    console.error('週間予測エラー:', error);
                    alert('週間予測の取得中にエラーが発生しました: ' + error.message);
                }
            }
        } catch (error) {
            console.error('予測実行エラー:', error);
            alert('予測の実行中にエラーが発生しました: ' + error.message);
        }
    });
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

// フォームデータの準備
const formData = {
    date: document.getElementById('prediction-date').value,
    total_outpatient: parseInt(document.getElementById('total-outpatient').value),
    intro_outpatient: parseInt(document.getElementById('intro-outpatient').value),
    er_patients: parseInt(document.getElementById('er-patients').value),
    bed_count: parseInt(document.getElementById('bed-count').value),
    y: parseInt(document.getElementById('y').value)  // 前日の新規入院患者数
};

// バリデーション
if (!formData.date || isNaN(formData.total_outpatient) || isNaN(formData.intro_outpatient) || 
    isNaN(formData.er_patients) || isNaN(formData.bed_count) || isNaN(formData.y)) {
    alert('すべての項目を入力してください');
    return;
}

// 値の範囲チェック
if (formData.total_outpatient < 0 || formData.total_outpatient > 2000) {
    alert('外来患者数は0から2000の間で入力してください');
    return;
}
if (formData.intro_outpatient < 0 || formData.intro_outpatient > 200) {
    alert('紹介患者数は0から200の間で入力してください');
    return;
}
if (formData.er_patients < 0 || formData.er_patients > 100) {
    alert('救急搬送患者数は0から100の間で入力してください');
    return;
}
if (formData.bed_count < 0 || formData.bed_count > 1000) {
    alert('入院患者数は0から1000の間で入力してください');
    return;
}
if (formData.y < 0 || formData.y > 100) {
    alert('前日の新規入院患者数は0から100の間で入力してください');
    return;
} 