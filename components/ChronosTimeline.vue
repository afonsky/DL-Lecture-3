<template>
  <div ref="containerRef" class="chronos-timeline-wrapper"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { ChronosTimeline, attachChronosStyles } from 'chronos-timeline-md'
import type { ChronosPluginSettings } from 'chronos-timeline-md'

const props = withDefaults(defineProps<{
  source: string
  settings?: Partial<ChronosPluginSettings>
}>(), {
  source: '',
  settings: () => ({})
})

const containerRef = ref<HTMLElement | null>(null)
let timeline: ChronosTimeline | null = null

const defaultSettings: ChronosPluginSettings = {
  selectedLocale: 'en',
  align: 'left',
  clickToUse: false,
  roundRanges: false,
  useUtc: false,
  useAI: false,
}

const renderTimeline = () => {
  if (!containerRef.value) return
  
  // Destroy existing timeline if any
  if (timeline) {
    timeline.destroy()
    timeline = null
  }
  
  // Clear container
  containerRef.value.innerHTML = ''
  
  if (props.source.trim()) {
    timeline = ChronosTimeline.render(
      containerRef.value,
      props.source,
      { ...defaultSettings, ...props.settings }
    )
  }
}

onMounted(() => {
  attachChronosStyles(document)
  nextTick(renderTimeline)
})

onBeforeUnmount(() => {
  if (timeline) {
    timeline.destroy()
    timeline = null
  }
})

watch(() => props.source, () => nextTick(renderTimeline))
watch(() => props.settings, () => nextTick(renderTimeline), { deep: true })
</script>

<style scoped>
.chronos-timeline-wrapper {
  width: 100%;
  min-height: 200px;
}
</style>
