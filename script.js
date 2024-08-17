document.addEventListener('DOMContentLoaded', () => {
    const toggles = document.querySelectorAll('.toggle');

    toggles.forEach(toggle => {
        toggle.addEventListener('click', (event) => {
            event.preventDefault();
            const nested = toggle.nextElementSibling;

            if (nested.style.display === 'block') {
                nested.style.display = 'none';
            } else {
                nested.style.display = 'block';
            }
        });
    });
});

