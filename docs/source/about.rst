About
=====

``zea`` is a toolbox intended to support research towards cognitive ultrasound imaging, a concept described in :cite:t:`about-van2024active`. The idea is to close the action-perception loop in ultrasound imaging, where acquisition and reconstruction are tightly coupled to tackle some of the persistent challenges in the field of ultrasound imaging.

While the full realization of cognitive ultrasound imaging remains an ongoing effort, we hope this toolbox will help spur further research and development in the field.

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 3em 0;">
     <!-- Dark mode image -->
     <img
       src="_static/zea_perception_action-Dark.svg"
       alt="Closing the action-perception loop in ultrasound imaging"
       style="display: none; width: 80%; padding-bottom: 1em;"
       class="only-dark"
     />
     <!-- Light mode image -->
     <img
       src="_static/zea_perception_action-Light.svg"
       alt="Closing the action-perception loop in ultrasound imaging"
       style="display: none; width: 80%; padding-bottom: 1em;"
       class="only-light"
     />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
        High-level overview of an ultrasound perception-action loop implemented in zea.
     </div>
   </div>
   <style>
     @media (prefers-color-scheme: dark) {
       .only-dark { display: block !important; }
     }
     @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
       .only-light { display: block !important; }
     }
   </style>


.. note::
    **What's in a name?**

    It's just a name... If we have to give it some meaning: ``zea`` is derived from the scientific name for corn, *Zea mays*, a staple food crop. If you look at the logo, you can see that the kernels of the corn cob have some resemblance with either a sensing matrix or possibly the elements of an array. The high-dimensional and structured nature of the corn cob also reflects the complexity of ultrasound data.


Core maintainers
----------------

- `Tristan Stevens <https://github.com/tristan-deep>`_, Eindhoven University of Technology, The Netherlands
- `Wessel van Nierop <https://github.com/wesselvannierop>`_, Eindhoven University of Technology, The Netherlands
- `Ben Luijten <https://github.com/benluijten>`_, Eindhoven University of Technology, The Netherlands

Active contributors
-------------------

A list of active contributors can be found on the `GitHub contributors page <https://github.com/tue-bmd/zea/graphs/contributors>`_. If you would like to contribute, please see the :doc:`contributing` guide.


License
-------

This project is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.


Citation
--------

Please see the :ref:`citation` guide for citation information of `zea`.

Papers
------

The following list contains some of the papers that have been published using ``zea``. If you have used ``zea`` in your work, please consider adding it to the list by creating a pull request on GitHub. See the :doc:`contributing` guide for more information.


.. bibliography:: ../../paper/paper.bib
   :style: unsrt
   :keyprefix: about-
   :labelprefix: A-

   van2024active
   luijten2020adaptive
   van2024off
   nolan2024active
   stevens2024dehazing
   federici2024active
   stevens2025sequential
   stevens2025high
   stevens2025deep
   penninga2025deep
   van2025patient
   stevens2025semantic
   stevens2026nuclear
   nolan2026task
   federici2026informative
