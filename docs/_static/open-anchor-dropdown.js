/**
 * Auto-open a sphinx-design dropdown when navigating to its anchor.
 * Handles both page-load with a hash and same-page hash changes (↓ links).
 */
(function () {
    function openDropdownForHash(hash) {
        if (!hash) return;
        const id = decodeURIComponent(hash.startsWith('#') ? hash.slice(1) : hash);
        const target = document.getElementById(id);
        if (!target) return;

        // The label may resolve to the <details> itself, to a wrapper <div>
        // containing it, or to a preceding <span> anchor.
        let details =
            target.tagName === 'DETAILS'
                ? target
                : target.querySelector('details');

        if (!details) {
            // Sphinx sometimes renders the explicit label as a <span id="...">
            // that precedes the directive container in the DOM.
            let sib = target.nextElementSibling;
            while (sib && !details) {
                details =
                    sib.tagName === 'DETAILS' ? sib : sib.querySelector('details');
                sib = sib.nextElementSibling;
            }
        }

        if (details) details.open = true;
    }

    document.addEventListener('DOMContentLoaded', function () {
        openDropdownForHash(window.location.hash);
    });

    window.addEventListener('hashchange', function () {
        openDropdownForHash(window.location.hash);
    });
})();
