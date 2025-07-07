document.getElementById('imageInput').addEventListener('change', function (event) {
  const file = event.target.files[0];
  const preview = document.getElementById('preview');
  const analyzeBtn = document.getElementById('analyzeBtn');

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    analyzeBtn.disabled = false;
  } else {
    preview.src = '';
    preview.style.display = 'none';
    analyzeBtn.disabled = true;
  }
});




