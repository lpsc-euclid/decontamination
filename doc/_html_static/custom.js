/*--------------------------------------------------------------------------------------------------------------------*/

document.addEventListener('DOMContentLoaded', function() {

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('li.right').forEach(function(item) {

        if(item.querySelector('a[href="py-modindex.html"]'))
        {
            item.style.display = 'none';
        }
    });

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('.reference .pre').forEach((item) => {

        item.textContent = item.textContent.split('.').pop();
    });

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('dt').forEach(function(dt) {

        if(dt.textContent.includes('Attributes'))
        {
            const dd = dt.nextElementSibling;

            if(dd && dd.nodeName === 'DD')
            {
                dd.remove();
                dt.remove();
            }
        }
    });

    /*----------------------------------------------------------------------------------------------------------------*/
});

/*--------------------------------------------------------------------------------------------------------------------*/
