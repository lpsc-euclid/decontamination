/*--------------------------------------------------------------------------------------------------------------------*/

document.addEventListener('DOMContentLoaded', function() {

    const tocItems = document.querySelectorAll('.reference .pre');

    tocItems.forEach((item) => {

        item.textContent = item.textContent.split('.').pop();
    });
});

/*--------------------------------------------------------------------------------------------------------------------*/
