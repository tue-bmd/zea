/**
 * Make sphinx-design tabs and dropdowns deep-linkable.
 *
 * When navigating to an anchor (on page load or via a same-page hash change)
 * this:
 *   1. selects the tab if the anchor is a tab label (e.g. ``#group-metadata``);
 *   2. selects any enclosing tab(s) so a target nested inside a tab panel is
 *      revealed (e.g. ``#spec-meta-subject`` lives in the ``metadata`` tab);
 *   3. opens the target dropdown (or a dropdown following the anchor span).
 */
(function () {
    var TAB_STORAGE_PREFIX = 'sphinx-design-tab-id-';

    function activateTab(label) {
        // ``label`` is a <label class="sd-tab-label"> immediately preceded by
        // its radio <input>. Checking that input selects the tab.
        var input = label.previousElementSibling;
        if (!input || input.tagName !== 'INPUT') return;
        input.checked = true;

        // Keep sphinx-design's session-storage sync in agreement, otherwise it
        // may re-select a different tab on the next page load.
        var group = label.getAttribute('data-sync-group');
        var id = label.getAttribute('data-sync-id');
        if (group && id) {
            try {
                window.sessionStorage.setItem(TAB_STORAGE_PREFIX + group, id);
            } catch (e) {
                /* sessionStorage may be unavailable; selection still works */
            }
        }
    }

    function activateAncestorTabs(el) {
        // Walk up through any enclosing tab panels, selecting each so that
        // ``el`` becomes visible. Supports nested tab-sets.
        var content = el.closest ? el.closest('.sd-tab-content') : null;
        while (content) {
            var label = content.previousElementSibling; // <label> precedes content
            if (label && label.classList.contains('sd-tab-label')) {
                activateTab(label);
            }
            content = content.parentElement
                ? content.parentElement.closest('.sd-tab-content')
                : null;
        }
    }

    function openDropdown(target) {
        // The label may resolve to the <details> itself, to a wrapper <div>
        // containing it, or to a preceding <span> anchor.
        var details =
            target.tagName === 'DETAILS' ? target : target.querySelector('details');

        if (!details) {
            // Sphinx sometimes renders the explicit label as a <span id="...">
            // that precedes the directive container in the DOM.
            var sib = target.nextElementSibling;
            while (sib && !details) {
                details =
                    sib.tagName === 'DETAILS' ? sib : sib.querySelector('details');
                sib = sib.nextElementSibling;
            }
        }

        if (details) details.open = true;
    }

    function handleHash(hash) {
        if (!hash) return;
        var id = decodeURIComponent(hash.charAt(0) === '#' ? hash.slice(1) : hash);
        var target = document.getElementById(id);
        if (!target) return;

        if (target.classList.contains('sd-tab-label')) {
            // The anchor is a tab itself — select it (and any outer tabs).
            activateTab(target);
            activateAncestorTabs(target);
        } else {
            // Reveal the target if it lives inside one or more tab panels,
            // then open it if it is (or precedes) a dropdown.
            activateAncestorTabs(target);
            openDropdown(target);
        }

        // Re-scroll now that tabs/dropdowns have expanded.
        if (target.scrollIntoView) target.scrollIntoView();
    }

    function updateHash(id) {
        // Reflect the current tab in the address bar so it can be copied and
        // shared, without scrolling or polluting the back/forward history.
        var newHash = '#' + id;
        if (window.location.hash === newHash) return;
        if (window.history && window.history.replaceState) {
            window.history.replaceState(null, '', newHash);
        } else {
            window.location.hash = id;
        }
    }

    function wireTabHashSync() {
        // When a named tab is selected, update the URL to its anchor.
        document.querySelectorAll('label.sd-tab-label[id]').forEach(function (label) {
            label.addEventListener('click', function () {
                updateHash(label.id);
            });
        });
    }

    function accordionGroupOf(details) {
        // A dropdown belongs to an accordion group if it carries a
        // ``dropdown-accordion-<group>`` class (set via :class-container:).
        for (var i = 0; i < details.classList.length; i++) {
            if (details.classList[i].indexOf('dropdown-accordion-') === 0) {
                return details.classList[i];
            }
        }
        return null;
    }

    function wireDropdownAccordions() {
        // Make grouped dropdowns behave like tabs: opening one closes the
        // others in its group and updates the URL to the open panel's anchor.
        var groups = {};
        document
            .querySelectorAll('details[class*="dropdown-accordion-"]')
            .forEach(function (details) {
                var group = accordionGroupOf(details);
                if (!group) return;
                (groups[group] = groups[group] || []).push(details);
            });

        Object.keys(groups).forEach(function (group) {
            var members = groups[group];
            members.forEach(function (details) {
                details.addEventListener('toggle', function () {
                    if (!details.open) return;
                    members.forEach(function (other) {
                        if (other !== details && other.open) other.open = false;
                    });
                    if (details.id) updateHash(details.id);
                });
            });
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        wireTabHashSync();
        wireDropdownAccordions();
        // Defer so this runs after sphinx-design's own tab session-restore
        // (both register DOMContentLoaded listeners).
        window.setTimeout(function () {
            handleHash(window.location.hash);
        }, 0);
    });

    window.addEventListener('hashchange', function () {
        handleHash(window.location.hash);
    });
})();
